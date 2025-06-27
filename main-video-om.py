import cv2
import numpy as np
import acl
import time
import ctypes


class YOLOv11_NPU_ZeroCopy:
    def __init__(self, model_path):
        print("[初始化] 开始初始化NPU引擎...")
        # 1. 初始化ACL环境
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"[错误] ACL初始化失败，错误码: {ret}")
        print("[初始化] ACL初始化成功")

        # 2. 设置设备
        self.device_id = 0
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            acl.finalize()
            raise RuntimeError(f"[错误] 设置设备失败，错误码: {ret}")
        print(f"[初始化] 设备 {self.device_id} 设置成功")

        # 3. 加载OM模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"[错误] 模型加载失败，错误码: {ret}")
        print(f"[初始化] 模型加载成功: {model_path}")

        # 4. 创建模型描述
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"[错误] 获取模型描述失败，错误码: {ret}")
        print("[初始化] 模型描述创建成功")

        # 5. 获取输入输出尺寸
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        print(f"[初始化] 输入尺寸: {self.input_size}字节, 输出尺寸: {self.output_size}字节")

        # 6. 分配输入输出内存（兼容性修复）
        # 使用直接分配替代mem_pool_create（解决ACL库兼容性问题）
        self.input_buffer, ret = acl.rt.malloc(self.input_size, 0)  # 0=ACL_MEM_MALLOC_HUGE_FIRST
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"[错误] 输入内存分配失败，错误码: {ret}")

        self.output_buffer, ret = acl.rt.malloc(self.output_size, 0)
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"[错误] 输出内存分配失败，错误码: {ret}")
        print(f"[初始化] 内存分配完成 (输入: {hex(self.input_buffer)}, 输出: {hex(self.output_buffer)})")

        # 7. 创建异步流
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"[错误] 创建流失败，错误码: {ret}")
        print("[初始化] NPU引擎初始化完成（零拷贝模式）")

    def _release_resources(self):
        """统一释放资源"""
        print("[清理] 开始释放NPU资源...")
        if hasattr(self, 'input_buffer') and self.input_buffer:
            acl.rt.free(self.input_buffer)
            print("[清理] 输入内存已释放")
        if hasattr(self, 'output_buffer') and self.output_buffer:
            acl.rt.free(self.output_buffer)
            print("[清理] 输出内存已释放")
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
            print("[清理] 模型已卸载")
        if hasattr(self, 'stream'):
            acl.rt.destroy_stream(self.stream)
            print("[清理] 流已销毁")
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[清理] NPU资源释放完成")

    def preprocess(self, frame):
        """零拷贝预处理：调整尺寸、填充、格式转换"""
        if frame is None or frame.size == 0:
            raise ValueError("[预处理] 输入帧为空")

        # 1. 调整尺寸（保持宽高比）
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (new_w, new_h))
        print(f"[预处理] 尺寸调整: {w}x{h} -> {new_w}x{new_h}, 缩放比例: {scale:.2f}")

        # 2. 填充至640x640
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        print(f"[预处理] 图像填充: 水平 {pad_x}px, 垂直 {pad_y}px")

        # 3. 转换为模型输入格式
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # [1,3,640,640]

        # 4. 确保内存对齐
        if img.ctypes.data % 32 != 0:
            img = np.require(img, requirements=['C_CONTIGUOUS', 'ALIGNED'])
            print("[预处理] 内存对齐调整完成")

        # 5. 零拷贝内存映射
        img_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = acl.rt.memcpy(
            self.input_buffer, 0,
            ctypes.cast(img_ptr, ctypes.c_void_p),
            img.nbytes,
            acl.rt.memcpy_kind.MEMCPY_HOST_TO_DEVICE
        )
        if ret != 0:
            raise RuntimeError(f"[错误] 内存映射失败，错误码: {ret}")

        # 6. 等待映射完成
        acl.rt.synchronize_stream(self.stream)
        print("[预处理] 零拷贝内存映射完成")

        # 保存预处理参数用于后处理
        self.scale = scale
        self.padding = (pad_x, pad_y)
        return img.shape[2:]

    def infer_sync(self):
        """同步推理（阻塞式）"""
        print("[推理] 开始推理...")
        start_time = time.time()

        # 1. 创建数据集缓冲区
        input_dataset = acl.mdl.create_dataset_buffer(self.input_buffer)
        output_dataset = acl.mdl.create_dataset_buffer(self.output_buffer)

        # 2. 执行推理
        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        if ret != 0:
            raise RuntimeError(f"[错误] 推理执行失败，错误码: {ret}")

        # 3. 等待推理完成
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != 0:
            raise RuntimeError(f"[错误] 流同步失败，错误码: {ret}")

        # 4. 准备输出缓冲区
        host_output = np.zeros((1, 6, 8400), dtype=np.float32)

        # 5. 零拷贝输出映射
        ret = acl.rt.memcpy(
            host_output.ctypes.data, 0,
            self.output_buffer, 0,
            self.output_size,
            acl.rt.memcpy_kind.MEMCPY_DEVICE_TO_HOST
        )
        if ret != 0:
            raise RuntimeError(f"[错误] 输出拷贝失败，错误码: {ret}")

        acl.rt.synchronize_stream(self.stream)

        infer_time = (time.time() - start_time) * 1000
        print(f"[推理] 推理完成! 耗时: {infer_time:.2f}ms")
        return host_output

    def decode_predictions(self, pred, conf_thres=0.5):
        """解码NPU输出为边界框"""
        print("[后处理] 开始解码预测结果...")
        pred = pred[0]  # [6,8400]

        # 1. 置信度过滤
        keep_mask = pred[4] > conf_thres
        cx, cy, w, h, conf, cls_prob = pred[:6, keep_mask]

        # 2. 计算边界框坐标
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5

        # 3. 类别判断
        cls_id = (cls_prob > 0.5).astype(int)

        print(f"[后处理] 检测到 {len(x1)} 个候选目标")
        return np.column_stack([x1, y1, x2, y2, conf, cls_id])

    def diou_nms(self, boxes, iou_thres=0.45):
        """DIoU非极大值抑制"""
        if len(boxes) == 0:
            return []

        print(f"[NMS] 开始DIoU-NMS处理，候选框数量: {len(boxes)}")
        start_time = time.time()

        # 1. 按置信度排序
        idxs = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[idxs]

        # 2. DIoU-NMS处理
        keep = []
        while len(boxes) > 0:
            keep.append(boxes[0])
            if len(boxes) == 1:
                break

            # 计算DIoU（向量化）
            ious = np.array([self.calculate_diou(boxes[0], box) for box in boxes[1:]])
            mask = ious < iou_thres
            boxes = boxes[1:][mask]

        nms_time = (time.time() - start_time) * 1000
        print(f"[NMS] 处理完成! 保留 {len(keep)} 个目标，耗时: {nms_time:.2f}ms")
        return np.array(keep)

    def calculate_diou(self, box1, box2):
        """计算DIoU值"""
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
        """将边界框坐标转换回原始图像空间"""
        if len(boxes) == 0:
            return boxes

        print(f"[坐标转换] 开始转换 {len(boxes)} 个边界框坐标...")
        h_orig, w_orig = orig_shape[:2]
        pad_x, pad_y = self.padding

        # 应用逆变换
        boxes[:, 0] = (boxes[:, 0] - pad_x) / self.scale
        boxes[:, 1] = (boxes[:, 1] - pad_y) / self.scale
        boxes[:, 2] = (boxes[:, 2] - pad_x) / self.scale
        boxes[:, 3] = (boxes[:, 3] - pad_y) / self.scale

        # 裁剪到图像边界
        np.clip(boxes[:, 0], 0, w_orig, out=boxes[:, 0])
        np.clip(boxes[:, 1], 0, h_orig, out=boxes[:, 1])
        np.clip(boxes[:, 2], 0, w_orig, out=boxes[:, 2])
        np.clip(boxes[:, 3], 0, h_orig, out=boxes[:, 3])

        return boxes

    def visualize_detections(self, frame, boxes):
        """在帧上可视化检测结果"""
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            label = f"{cls_id}:{conf:.2f}"

            # 绘制边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 绘制标签背景
            cv2.rectangle(frame, (int(x1), int(y1) - 20),
                          (int(x1) + len(label) * 10, int(y1)), (0, 255, 0), -1)

            # 绘制标签文本
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return frame

    def release(self):
        """释放NPU资源"""
        self._release_resources()


# ===== 主处理流程 =====
if __name__ == "__main__":
    # 配置路径
    video_path = "./datasets/video/cat_blibli_7.mp4"
    model_path = "./runs/train/train/weights/best.om"
    output_video_path = "output_detection.mp4"

    # 初始化关键变量（解决NameError问题）
    frame_count = 0
    cap = out = infer_engine = None

    try:
        # 初始化NPU推理引擎
        print("\n" + "=" * 50)
        print(" 香橙派 AIpro 视频目标检测系统启动 ")
        print("=" * 50)
        infer_engine = YOLOv11_NPU_ZeroCopy(model_path)

        # 打开视频文件
        print("\n[视频处理] 打开视频文件...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[错误] 无法打开视频: {video_path}")

        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[视频信息] 分辨率: {frame_width}x{frame_height}, FPS: {fps:.1f}, 总帧数: {total_frames}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        print(f"[视频处理] 输出文件: {output_video_path}")

        # 性能监控
        start_time = time.time()
        last_log_time = time.time()

        print("\n[视频处理] 开始处理视频...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[视频处理] 视频读取完成")
                break

            frame_count += 1
            orig_shape = frame.shape

            try:
                print(f"\n[帧处理] 开始处理第 {frame_count}/{total_frames} 帧")
                frame_start = time.time()

                # 1. 预处理
                infer_engine.preprocess(frame)

                # 2. NPU推理
                output = infer_engine.infer_sync()

                # 3. 后处理
                boxes = infer_engine.decode_predictions(output, conf_thres=0.5)
                if len(boxes) > 0:
                    boxes = infer_engine.diou_nms(boxes)
                    boxes = infer_engine.transform_boxes(boxes, orig_shape)
                    print(f"[检测结果] 有效目标数量: {len(boxes)}")

                # 4. 可视化
                frame = infer_engine.visualize_detections(frame, boxes)

                # 5. 显示帧率和进度
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                progress = frame_count / total_frames * 100

                # 绘制统计信息
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"进度: {progress:.1f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"目标: {len(boxes)}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 6. 写入输出视频
                out.write(frame)

                # 7. 显示处理结果
                cv2.imshow('YOLOv11 NPU Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[用户中断] 检测到用户中断请求")
                    break

                # 每5秒打印一次进度
                if time.time() - last_log_time > 5:
                    avg_fps = frame_count / (time.time() - start_time)
                    print(f"[进度] 已处理 {frame_count}/{total_frames} 帧 ({progress:.1f}%), 平均FPS: {avg_fps:.1f}")
                    last_log_time = time.time()

                # 帧处理时间统计
                frame_time = (time.time() - frame_start) * 1000
                print(f"[帧处理] 完成! 耗时: {frame_time:.2f}ms")

            except Exception as e:
                print(f"[警告] 处理第 {frame_count} 帧时出错: {str(e)}")
                # 跳过当前帧继续处理
                continue

    except Exception as e:
        print(f"[严重错误] 程序运行出错: {str(e)}")
    finally:
        print("\n" + "=" * 50)
        print(" 系统关闭中，释放资源... ")
        print("=" * 50)
        if infer_engine is not None:
            infer_engine.release()
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"[统计] 处理完成! 总帧数: {frame_count}, 总耗时: {total_time:.1f}秒, 平均FPS: {avg_fps:.1f}")
            print(f"[输出] 检测结果已保存至: {output_video_path}")
        print("系统已安全关闭")