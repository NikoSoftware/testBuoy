import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time


class YOLOv11_NPU_Inference:
    def __init__(self, model_path):
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)
        self.output_shape = (1, 6, 8400)
        self.conf_thres = 0  # 初始阈值设为极低值
        print(f"[NPU] Model loaded. Input shape: {self.input_shape}")

        # 调试日志文件
        self.debug_log = open("np_debug.log", "w")

    def __del__(self):
        self.debug_log.close()

    def preprocess(self, frame):
        """图像预处理：保持宽高比的缩放和填充"""
        h, w = frame.shape[:2]
        scale = min(self.input_shape[1] / w, self.input_shape[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 关键修复：使用YOLO标准填充值114 [7](@ref)
        canvas = np.full((*self.input_shape, 3), 114, dtype=np.uint8)
        top = (self.input_shape[0] - new_h) // 2
        left = (self.input_shape[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC->NCHW
        print(f"[Preprocess] {h}x{w} -> {new_h}x{new_w} | Scale: {scale:.4f}")
        return blob, (left, top, scale), (h, w)

    def postprocess(self, outputs, meta, conf_thres=0.01, iou_thres=0.45):
        """重构后处理逻辑 - 针对绝对坐标输出"""
        left, top, scale = meta["padding"]
        orig_h, orig_w = meta["original_shape"]

        # 保存原始输出用于调试
        np.save(f"np_output_{time.time()}.npy", outputs)

        # 解析输出结构 [1,6,8400] -> [8400,6]
        # 维度含义: [batch, channels, num_predictions]
        preds = outputs[0][0].T  # 转置为[8400,6]

        boxes, confidences, class_ids = [], [], []

        for i in range(preds.shape[0]):
            # 关键修复：直接使用绝对坐标 [3,7](@ref)
            cx, cy, w, h, obj_conf, cls_prob = preds[i]

            # 单类别模型：置信度=目标置信度 [7](@ref)
            conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid激活

            if conf < conf_thres:
                continue

            # 计算边界框坐标（绝对坐标直接使用）
            x1 = int((cx - w / 2 - left) / scale)
            y1 = int((cy - h / 2 - top) / scale)
            x2 = int((cx + w / 2 - left) / scale)
            y2 = int((cy + h / 2 - top) / scale)

            # 边界检查
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            # 过滤无效框
            bbox_w, bbox_h = x2 - x1, y2 - y1
            if bbox_w < 2 or bbox_h < 2:
                continue

            # 存储结果（单类别模型class_id=0）
            boxes.append([x1, y1, bbox_w, bbox_h])
            confidences.append(float(conf))
            class_ids.append(0)

            # 调试日志
            self.debug_log.write(f"Pred{i}: conf={conf:.4f} box=({x1},{y1},{x2},{y2})\n")

        # NMS过滤 (使用OpenCV 4.x兼容方式) [3](@ref)
        detections = []
        if confidences:
            # 转换boxes格式为[x,y,w,h]
            boxes_np = np.array(boxes)

            # 关键修复：使用正确NMS参数
            indices = cv2.dnn.NMSBoxes(
                boxes_np.tolist(),
                confidences,
                conf_thres,  # 置信度阈值
                iou_thres  # NMS阈值
            )

            if indices is not None:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections.append([x, y, x + w, y + h, confidences[i], class_ids[i]])

        print(f"[Postprocess] Candidates: {len(confidences)} → NMS: {len(detections)}")
        return detections

    def run_video(self, video_path, output_path="./output_dog.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        # 获取视频属性
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_detection_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            print(f"\n--- Frame {frame_idx}/{total_frames} ---")

            # 1. 预处理
            start_time = time.time()
            blob, padding_info, orig_shape = self.preprocess(frame)
            preprocess_time = time.time() - start_time

            # 2. NPU推理
            start_time = time.time()
            outputs = self.session.infer([blob])
            infer_time = time.time() - start_time

            # 调试信息
            print(f"[NPU] Shape: {outputs[0].shape} | Range: {outputs[0].min():.2f}-{outputs[0].max():.2f}")
            print(f"[Time] Preprocess: {preprocess_time:.4f}s | Inference: {infer_time:.4f}s")

            # 3. 后处理
            start_time = time.time()
            dets = self.postprocess(
                outputs,
                {"padding": padding_info, "original_shape": orig_shape},
                conf_thres=self.conf_thres
            )
            postprocess_time = time.time() - start_time
            print(f"[Time] Postprocess: {postprocess_time:.4f}s")

            # 动态调整阈值策略
            current_time = time.time()
            if len(dets) == 0:
                # 超过5秒无检测则降低阈值
                if current_time - last_detection_time > 5:
                    self.conf_thres = max(0.001, self.conf_thres * 0.7)
                    print(f"[Threshold] Adjusting to {self.conf_thres:.4f}")
            else:
                last_detection_time = current_time

            # 4. 结果可视化
            for x1, y1, x2, y2, conf, cls_id in dets:
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f"{cls_id}:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print(f"[Detection] {label} at [{x1},{y1},{x2},{y2}]")

            # 显示处理信息
            info_text = f"FPS: {1 / (preprocess_time + infer_time + postprocess_time):.1f} | Thres: {self.conf_thres:.4f}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 写入输出帧
            out.write(frame)


        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[Output] Saved to {output_path}")


if __name__ == "__main__":
    model_path = "./runs/train/train/weights/best.om"
    video_path = "./datasets/video/dog.mp4"

    print("Starting YOLOv11 NPU inference...")
    detector = YOLOv11_NPU_Inference(model_path)
    detector.run_video(video_path)
    print("Execution completed")