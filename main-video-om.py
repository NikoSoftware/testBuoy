import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


class YOLOv11_NPU_Inference:
    def __init__(self, model_path):
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)
        self.output_shape = (1, 6, 8400)
        self.conf_thres = 0.01  # 大幅降低初始阈值
        print(f"NPU model loaded. Input shape: {self.input_shape}")

    def preprocess(self, frame):
        """图像预处理：保持宽高比的缩放和填充"""
        h, w = frame.shape[:2]
        scale = min(self.input_shape[1] / w, self.input_shape[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.full((*self.input_shape, 3), 128, dtype=np.uint8)
        top = (self.input_shape[0] - new_h) // 2
        left = (self.input_shape[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC->NCHW
        print(f"Preprocessed: {h}x{w} -> {new_h}x{new_w}")
        return blob, (left, top, scale), (h, w)

    def postprocess(self, outputs, meta, conf_thres=0.01, iou_thres=0.45):
        """完全重构的后处理逻辑"""
        left, top, scale = meta["padding"]
        orig_h, orig_w = meta["original_shape"]

        # 关键修复1：正确解析输出结构 [1,6,8400]->[8400,6]
        preds = outputs[0][0].T  # 直接取第一个batch

        boxes, confidences, class_ids = [], [], []

        for i in range(preds.shape[0]):
            cx, cy, w, h, obj_conf, cls_conf = preds[i]

            # 关键修复2：Sigmoid激活置信度 [6](@ref)
            obj_conf = 1 / (1 + np.exp(-obj_conf))
            cls_conf = 1 / (1 + np.exp(-cls_conf))
            conf = obj_conf * cls_conf

            if conf < conf_thres:
                continue

            # 关键修复3：坐标归一化（绝对坐标转相对坐标）
            cx, cy, w, h = (val / 640 for val in (cx, cy, w, h))

            # 计算原图坐标
            x1 = int((cx - w / 2 - left) / scale)
            y1 = int((cy - h / 2 - top) / scale)
            x2 = int((cx + w / 2 - left) / scale)
            y2 = int((cy + h / 2 - top) / scale)

            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)

            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(conf))
            class_ids.append(0)  # 单类别模型

        # 关键修复4：正确配置NMS参数 [9](@ref)
        if confidences:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences,
                score_threshold=conf_thres,  # 置信度阈值
                nms_threshold=iou_thres  # IoU阈值
            )
        else:
            indices = []

        detections = []
        if isinstance(indices, np.ndarray) and indices.size > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append([x, y, x + w, y + h, confidences[i], class_ids[i]])

        print(f"Raw candidates: {len(confidences)} | After NMS: {len(detections)}")
        return detections

    def run_video(self, video_path, output_path="./output_dog.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            print(f"\n--- Frame {frame_idx} ---")

            # 预处理 -> NPU推理
            blob, meta, orig_shape = self.preprocess(frame)
            outputs = self.session.infer([blob])

            # 调试信息
            print(f"Output range: {outputs[0].min():.2f}-{outputs[0].max():.2f}")

            # 动态调整阈值：当连续5帧无检测时降低阈值
            if frame_idx % 5 == 0 and not hasattr(self, 'last_detection_count'):
                self.conf_thres = max(0.001, self.conf_thres * 0.7)
                print(f"Lowering threshold to {self.conf_thres:.4f}")

            # 后处理
            dets = self.postprocess(outputs, {
                "padding": meta,
                "original_shape": orig_shape
            }, conf_thres=self.conf_thres)

            # 更新检测计数器
            self.last_detection_count = len(dets)

            # 绘制结果
            for x1, y1, x2, y2, conf, cls_id in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_id}:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Detected: {label} at [{x1},{y1},{x2},{y2}]")

            out.write(frame)


        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    model_path = "./runs/train/train/weights/best.om"
    video_path = "./datasets/video/dog.mp4"

    print("Starting YOLOv11 NPU inference...")
    detector = YOLOv11_NPU_Inference(model_path)
    detector.run_video(video_path)
    print("Execution completed")