import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


class YOLOv11_NPU_Inference:
    def __init__(self, model_path):
        # 初始化NPU推理会话
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)  # YOLO标准输入尺寸
        self.output_shape = (1, 6, 8400)  # 指定输出张量形状

    def preprocess(self, frame):
        """图像预处理：缩放到输入尺寸并归一化"""
        # 保持宽高比的缩放
        h, w = frame.shape[:2]
        scale = min(self.input_shape[1] / w, self.input_shape[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 创建填充后的画布
        canvas = np.full((self.input_shape[0], self.input_shape[1], 3), 114, dtype=np.uint8)
        top = (self.input_shape[0] - new_h) // 2
        left = (self.input_shape[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        # 转换为RGB、归一化、调整维度顺序
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        return blob, (left, top, scale), (h, w)

    def postprocess(self, outputs, meta, conf_thres=0.5):
        """解析NPU输出：过滤检测结果并映射回原图坐标"""
        left, top, scale = meta["padding"]
        original_h, original_w = meta["original_shape"]

        # 提取预测结果 [1,6,8400] -> [8400,6]
        predictions = outputs[0].squeeze(0).T
        detections = []

        for pred in predictions:
            conf = pred[4]
            if conf < conf_thres:
                continue

            # 解析边界框 (cxcywh格式)
            cx, cy, w, h = pred[:4] * 2 - 0.5
            x1 = int((cx - w / 2 - left) / scale)
            y1 = int((cy - h / 2 - top) / scale)
            x2 = int((cx + w / 2 - left) / scale)
            y2 = int((cy + h / 2 - top) / scale)

            # 裁剪到图像范围内
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            # 添加结果 (x1, y1, x2, y2, conf, cls)
            detections.append([x1, y1, x2, y2, conf, int(pred[5])])

        return detections

    def run_video(self, video_path, output_path="./output.mp4"):
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

        # 帧处理循环
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理 -> NPU推理 -> 后处理
            blob, padding_info, orig_shape = self.preprocess(frame)
            outputs = self.session.infer([blob])  # 同步推理
            dets = self.postprocess(outputs, {
                "padding": padding_info,
                "original_shape": orig_shape
            })

            # 绘制检测框并输出
            for x1, y1, x2, y2, conf, cls_id in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_id}:{conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

            out.write(frame)

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 配置路径
    model_path = "./runs/train/train/weights/best.om"
    video_path = "./datasets/video/cat_blibli_7.mp4"

    # 初始化并运行
    detector = YOLOv11_NPU_Inference(model_path)
    detector.run_video(video_path)