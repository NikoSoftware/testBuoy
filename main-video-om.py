import cv2
import numpy as np
import acl
import time
from threading import Thread
from queue import Queue


class YOLOv11_NPU:
    def __init__(self, model_path):
        # 初始化NPU资源
        acl.init()
        self.device_id = 0
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)

        # 加载OM模型
        self.model_id, _ = acl.mdl.load_from_file(model_path)
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)

        # 3. 分配输入/输出内存 (修复后)
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

        # 使用整数策略替代枚举属性
        policy = 0  # ACL_MEM_MALLOC_HUGE_FIRST
        self.input_buffer, ret = acl.rt.malloc(self.input_size, policy)
        if ret != 0:
            raise RuntimeError(f"输入内存分配失败，错误码: {ret}")

        self.output_buffer, ret = acl.rt.malloc(self.output_size, policy)
        if ret != 0:
            acl.rt.free(self.input_buffer)  # 释放已分配的内存
            raise RuntimeError(f"输出内存分配失败，错误码: {ret}")

        # 创建推理流
        self.stream, _ = acl.rt.create_stream()

    def preprocess(self, frame):
        """视频帧预处理 - 优化硬件加速"""
        # 使用UMat实现零拷贝[6](@ref)
        if cv2.ocl.haveOpenCL():
            frame_umat = cv2.UMat(frame)
            img = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 缩放到640x640（保持宽高比）
        h, w = img.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        # 填充到640x640
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换为NPU输入格式
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # HWC→NCHW [1,3,640,640]
        img = np.ascontiguousarray(img)

        # 拷贝到NPU设备内存
        acl.rt.memcpy(self.input_buffer, 0, img.ctypes.data,
                      img.nbytes, acl.memcpy_type.HOST_TO_DEVICE)
        return img.shape[2:], scale, (pad_x, pad_y)

    def infer_sync(self):
        """同步推理（阻塞式）"""
        inputs = [acl.mdl.create_dataset_buffer(self.input_buffer)]
        outputs = [acl.mdl.create_dataset_buffer(self.output_buffer)]

        # 执行推理
        acl.mdl.execute(self.model_id, inputs, outputs)
        acl.rt.synchronize_stream(self.stream)

        # 取回输出数据 [1,6,8400]
        host_output = np.zeros((1, 6, 8400), dtype=np.float32)
        acl.rt.memcpy(host_output.ctypes.data, self.output_buffer,
                      self.output_size, acl.memcpy_type.DEVICE_TO_HOST)
        return host_output

    def decode_predictions(self, pred, conf_thres=0.5):
        """高效解码NPU输出"""
        pred = pred[0]  # [6,8400]
        # 向量化操作替代循环[3](@ref)
        cx, cy, w, h, conf, cls_prob = pred[:6]

        # 置信度过滤
        keep_mask = conf > conf_thres
        cx, cy, w, h, conf = [arr[keep_mask] for arr in [cx, cy, w, h, conf]]
        cls_id = (cls_prob[keep_mask] > 0.5).astype(int)

        # 计算边界框
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5

        boxes = np.column_stack([x1, y1, x2, y2, conf, cls_id])
        return boxes

    def diou_nms(self, boxes, iou_thres=0.45):
        """优化版DIoU-NMS"""
        if len(boxes) == 0:
            return []

        # 按置信度排序
        idxs = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[idxs]

        keep = []
        while len(boxes) > 0:
            keep.append(boxes[0])
            if len(boxes) == 1:
                break

            # 计算DIoU
            ious = np.array([self.calculate_diou(boxes[0], box) for box in boxes[1:]])

            # 保留IoU低于阈值的框
            mask = ious < iou_thres
            boxes = boxes[1:][mask]

        return np.array(keep)

    def calculate_diou(self, box1, box2):
        """向量化DIoU计算"""
        # 交集坐标
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])

        # 交集面积
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 各自面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        # IoU计算
        iou = inter_area / (union_area + 1e-7)

        # 中心点距离
        center_dist = ((box1[0] + box1[2]) / 2 - (box2[0] + box2[2]) / 2) ** 2 + \
                      ((box1[1] + box1[3]) / 2 - (box2[1] + box2[3]) / 2) ** 2

        # 最小闭包区域对角线
        diag_len = (max(box1[2], box2[2]) - min(box1[0], box2[0])) ** 2 + \
                   (max(box1[3], box2[3]) - min(box1[1], box2[1])) ** 2

        return iou - (center_dist / (diag_len + 1e-7))

    def transform_boxes(self, boxes, orig_shape, scale, padding):
        """坐标逆变换：640x640 → 原图尺寸"""
        h_orig, w_orig = orig_shape[:2]
        pad_x, pad_y = padding

        boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_x) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_y) / scale

        # 裁剪到图像边界内
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)

        return boxes

    def release(self):
        """释放NPU资源"""
        acl.rt.free(self.input_buffer)
        acl.rt.free(self.output_buffer)
        acl.mdl.unload(self.model_id)
        acl.rt.destroy_stream(self.stream)
        acl.rt.reset_device(self.device_id)
        acl.finalize()


# ================ 视频处理主流程 ================
class VideoProcessor:
    def __init__(self, video_path, model_path):
        # 优化视频解码[6](@ref)
        self.cap = cv2.VideoCapture()
        self.cap.open(video_path, apiPreference=cv2.CAP_FFMPEG, params=[
            cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY
        ])

        # 创建视频写入器
        self.writer = self.create_video_writer(video_path, "./img/output.mp4")

        # 初始化YOLOv11 NPU引擎
        self.infer_engine = YOLOv11_NPU(model_path)

        # 性能监控
        self.fps_counter = []
        self.prev_time = time.time()

        # 多线程队列
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)

        # 启动处理线程
        Thread(target=self.process_frames, daemon=True).start()

    def create_video_writer(self, input_path, output_path):
        """创建视频写入器"""
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def process_frames(self):
        """帧处理线程"""
        while True:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            if frame is None:
                break

            # 记录原始尺寸
            orig_shape = frame.shape

            # 预处理 & 获取缩放参数
            _, scale, padding = self.infer_engine.preprocess(frame)

            # NPU推理
            output = self.infer_engine.infer_sync()

            # 后处理
            boxes = self.infer_engine.decode_predictions(output, conf_thres=0.5)
            if len(boxes) > 0:
                boxes = self.infer_engine.diou_nms(boxes)
                boxes = self.infer_engine.transform_boxes(boxes, orig_shape, scale, padding)

            # 可视化
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_id}:{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 性能监控
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.fps_counter.append(fps)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(np.mean(self.fps_counter[-10:]))}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.result_queue.put(frame)

    def run(self):
        """主处理循环"""
        # 智能跳帧策略[6](@ref)
        target_fps = 30
        current_fps = self.cap.get(cv2.CAP_PROP_FPS)
        skip_ratio = max(1, int(current_fps / target_fps))

        frame_count = 0
        while True:
            # 跳帧处理
            for _ in range(skip_ratio - 1):
                self.cap.grab()
                frame_count += 1

            ret, frame = self.cap.retrieve()
            if not ret:
                break

            frame_count += 1
            print(f"处理帧: {frame_count}")

            # 添加到处理队列
            self.frame_queue.put(frame)

            # 获取处理结果
            if not self.result_queue.empty():
                result_frame = self.result_queue.get()
                cv2.imshow('YOLOv11M NPU', result_frame)
                self.writer.write(result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 清理资源
        self.frame_queue.put(None)
        self.cap.release()
        self.writer.release()
        self.infer_engine.release()
        cv2.destroyAllWindows()


# ================ 执行视频处理 ================
if __name__ == "__main__":
    video_path = "./datasets/video/cat_blibli_7.mp4"  # 替换为你的视频路径
    model_path = "./runs/train/train/weights/best.om"  # NPU模型路径

    processor = VideoProcessor(video_path, model_path)
    processor.run()