import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


class YOLOv11_NPU_Inference:
    def __init__(self, model_path):
        # 初始化NPU推理会话
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)  # YOLO标准输入尺寸
        self.output_shape = (1, 6, 8400)  # 指定输出张量形状

        # 添加调试信息[1,3](@ref)
        print("NPU model loaded successfully. Input shape:", self.input_shape)

    def preprocess(self, frame):
        """图像预处理：保持宽高比的缩放和填充"""
        h, w = frame.shape[:2]
        scale = min(self.input_shape[1] / w, self.input_shape[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 创建填充后的画布（使用128填充效果更好）[1](@ref)
        canvas = np.full((self.input_shape[0], self.input_shape[1], 3), 128, dtype=np.uint8)
        top = (self.input_shape[0] - new_h) // 2
        left = (self.input_shape[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        # 转换为RGB、归一化、调整维度顺序
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW

        # 添加调试信息
        print(f"Preprocessed: original({h},{w}) -> scaled({new_h},{new_w})")
        return blob, (left, top, scale), (h, w)

    def postprocess(self, outputs, meta, conf_thres=0.5, iou_thres=0.45):
        """解析NPU输出：过滤检测结果并映射回原图坐标"""
        left, top, scale = meta["padding"]
        original_h, original_w = meta["original_shape"]

        # 提取预测结果 [1,6,8400] -> [8400,6][9](@ref)
        predictions = outputs[0].squeeze(0).T

        # 关键修复：正确解析输出格式[9](@ref)
        # 6个值分别为: cx, cy, w, h, conf, class_prob
        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            cx, cy, w, h, conf, class_prob = pred

            # 跳过低置信度检测[3](@ref)
            if conf < conf_thres:
                continue

            # 关键修复：坐标转换（去归一化）[1](@ref)
            cx = cx * self.input_shape[1]
            cy = cy * self.input_shape[0]
            w = w * self.input_shape[1]
            h = h * self.input_shape[0]

            # 计算边界框坐标（从中心点转换为角点）
            x1 = int((cx - w / 2 - left) / scale)
            y1 = int((cy - h / 2 - top) / scale)
            x2 = int((cx + w / 2 - left) / scale)
            y2 = int((cy + h / 2 - top) / scale)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            # 计算框面积（用于后续过滤）
            area = (x2 - x1) * (y2 - y1)
            if area <= 10:  # 忽略过小的检测框[3](@ref)
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # 格式转为[x,y,width,height]
            confidences.append(float(conf))
            class_ids.append(int(class_prob))

        # 应用NMS过滤重叠框[3](@ref)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_thres, iou_thres
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append([
                    x, y, x + w, y + h, confidences[i], class_ids[i]
                ])

        print(f"Detected {len(detections)} objects after NMS")
        return detections

    def run_video(self, video_path, output_path="./output_dog.mp4"):
        """视频流推理主循环"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        # 获取视频属性并初始化输出
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0

        # 帧处理循环
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"\nProcessing frame {frame_count}")

            # 预处理 -> NPU推理 -> 后处理
            blob, padding_info, orig_shape = self.preprocess(frame)
            outputs = self.session.infer([blob])  # 同步推理

            # 添加输出调试信息[3](@ref)
            print(f"NPU output shape: {outputs[0].shape}")
            print(f"NPU output range: {outputs[0].min():.4f} - {outputs[0].max():.4f}")

            dets = self.postprocess(outputs, {
                "padding": padding_info,
                "original_shape": orig_shape
            })

            # 绘制检测框并输出
            for x1, y1, x2, y2, conf, cls_id in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_id}:{conf:.2f}"
                cv2.putText(frame, label,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)
                print(f"Detected object: {label} at [{x1},{y1},{x2},{y2}]")

            out.write(frame)



        # 释放资源
        cap.release()
        out.release()
        print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    # 配置路径
    model_path = "./runs/train/train/weights/best.om"
    video_path = "./datasets/video/dog.mp4"

    # 初始化并运行
    print("Starting YOLOv11 NPU inference...")
    detector = YOLOv11_NPU_Inference(model_path)
    detector.run_video(video_path)
    print("Execution completed")