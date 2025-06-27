import cv2
import numpy as np
import acl
import time


class YOLOv11_NPU:
    def __init__(self, model_path):
        # 初始化NPU资源
        ret = acl.init()
        assert ret == 0, f"ACL初始化失败: {ret}"

        self.device_id = 0
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"设置设备失败: {ret}"

        self.context, ret = acl.rt.create_context(self.device_id)
        assert ret == 0, f"创建上下文失败: {ret}"

        # 加载OM模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"模型加载失败: {ret}"

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == 0, f"获取模型描述失败: {ret}"

        # 分配输入输出内存（关键修复：使用整数策略）
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)

        # 内存分配策略：0 = ACL_MEM_MALLOC_HUGE_FIRST
        self.input_buffer, ret = acl.rt.malloc(self.input_size, 0)
        assert ret == 0, f"输入内存分配失败: {ret}"

        self.output_buffer, ret = acl.rt.malloc(self.output_size, 0)  # 输出[1,6,8400]
        assert ret == 0, f"输出内存分配失败: {ret}"

        # 创建推理流
        self.stream, ret = acl.rt.create_stream()
        assert ret == 0, f"创建流失败: {ret}"

    def preprocess(self, frame):
        """视频帧预处理 - 保持宽高比缩放并填充"""
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (new_w, new_h))

        # 填充灰色边框至640x640
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换为NPU输入格式
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # HWC→NCHW [1,3,640,640]
        img = np.ascontiguousarray(img)

        # 记录预处理参数
        self.scale = scale
        self.padding = (pad_x, pad_y)

        # 拷贝到NPU设备内存
        ret = acl.rt.memcpy(self.input_buffer, 0, img.ctypes.data,
                            img.nbytes, acl.rt.MEMCPY_HOST_TO_DEVICE)
        assert ret == 0, f"数据拷贝失败: {ret}"
        return img.shape[2:]  # 返回(640,640)

    def infer_sync(self):
        """同步推理（阻塞式）"""
        inputs = [acl.mdl.create_dataset_buffer(self.input_buffer)]
        outputs = [acl.mdl.create_dataset_buffer(self.output_buffer)]

        # 执行推理
        ret = acl.mdl.execute(self.model_id, inputs, outputs)
        assert ret == 0, f"推理执行失败: {ret}"

        # 阻塞等待完成
        ret = acl.rt.synchronize_stream(self.stream)
        assert ret == 0, f"同步流失败: {ret}"

        # 取回输出数据 [1,6,8400]
        host_output = np.zeros((1, 6, 8400), dtype=np.float32)
        ret = acl.rt.memcpy(host_output.ctypes.data, self.output_buffer,
                            self.output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)
        assert ret == 0, f"输出拷贝失败: {ret}"
        return host_output

    def decode_predictions(self, pred, conf_thres=0.5):
        """高效解码NPU输出（向量化操作）"""
        pred = pred[0]  # [6,8400]
        # 置信度过滤
        keep_mask = pred[4] > conf_thres
        cx, cy, w, h, conf, cls_prob = pred[:6, keep_mask]

        # 计算边界框
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        cls_id = (cls_prob > 0.5).astype(int)  # 二分类

        return np.column_stack([x1, y1, x2, y2, conf, cls_id])

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

            # 计算DIoU（向量化）
            ious = np.array([self.calculate_diou(boxes[0], box) for box in boxes[1:]])
            mask = ious < iou_thres
            boxes = boxes[1:][mask]

        return np.array(keep)

    def calculate_diou(self, box1, box2):
        """计算DIoU"""
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

    def transform_boxes(self, boxes, orig_shape):
        """坐标逆变换：640x640 → 原图尺寸"""
        h_orig, w_orig = orig_shape[:2]
        pad_x, pad_y = self.padding

        # 应用逆变换
        boxes[:, 0] = (boxes[:, 0] - pad_x) / self.scale  # x1
        boxes[:, 1] = (boxes[:, 1] - pad_y) / self.scale  # y1
        boxes[:, 2] = (boxes[:, 2] - pad_x) / self.scale  # x2
        boxes[:, 3] = (boxes[:, 3] - pad_y) / self.scale  # y2

        # 裁剪到图像边界
        np.clip(boxes[:, 0], 0, w_orig, out=boxes[:, 0])
        np.clip(boxes[:, 1], 0, h_orig, out=boxes[:, 1])
        np.clip(boxes[:, 2], 0, w_orig, out=boxes[:, 2])
        np.clip(boxes[:, 3], 0, h_orig, out=boxes[:, 3])
        return boxes

    def release(self):
        """释放NPU资源"""
        if hasattr(self, 'input_buffer'):
            acl.rt.free(self.input_buffer)
        if hasattr(self, 'output_buffer'):
            acl.rt.free(self.output_buffer)
        acl.mdl.unload(self.model_id)
        acl.rt.destroy_stream(self.stream)
        acl.rt.reset_device(self.device_id)
        acl.finalize()


# ===== 视频处理主流程 =====
if __name__ == "__main__":
    # 使用用户提供的路径
    video_path = "./datasets/video/cat_blibli_7.mp4"
    model_path = "./runs/train/train/weights/best.om"

    # 初始化NPU引擎
    infer_engine = YOLOv11_NPU(model_path)

    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入器（可选）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    # 性能监控
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        orig_shape = frame.shape

        # 1. 预处理
        infer_engine.preprocess(frame)

        # 2. NPU推理
        output = infer_engine.infer_sync()  # [1,6,8400]

        # 3. 后处理
        boxes = infer_engine.decode_predictions(output, conf_thres=0.5)
        if len(boxes) > 0:
            boxes = infer_engine.diou_nms(boxes)
            boxes = infer_engine.transform_boxes(boxes, orig_shape)

        # 4. 可视化
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            label = f"{cls_id}:{conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 5. 显示帧率
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 写入输出视频（可选）
        out.write(frame)

        # 显示结果
        cv2.imshow('YOLOv11 Video Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    infer_engine.release()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成! 总帧数: {frame_count}, 平均FPS: {fps:.1f}")