import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time  # 添加时间模块
import sys  # 用于进度条刷新


class YOLOv11_NPU_Inference:
    def __init__(self, model_path):
        # 初始化NPU推理会话
        self.session = InferSession(device_id=0, model_path=model_path)
        self.input_shape = (640, 640)  # YOLO标准输入尺寸
        self.output_shape = (1, 6, 8400)  # 指定输出张量形状
        self.total_inference_time = 0  # 总推理时间统计
        self.total_frames = 0  # 总帧数统计

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

    def run_video(self, video_path, output_path="./output.mp4", show_progress=True):
        """视频流推理主循环，添加进度和时间统计"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        # 获取视频属性并初始化输出
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"📺 开始视频处理: {video_path}")
        print(f"  - 分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total_frames}")

        if total_frames <= 0:
            print("⚠️ 警告: 无法确定视频总帧数，进度显示将受限")

        # 初始化性能指标
        frame_counter = 0
        processing_times = []
        start_time = time.time()

        # 帧处理循环
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            frame_start = time.time()

            # 预处理
            preprocess_start = time.time()
            blob, padding_info, orig_shape = self.preprocess(frame)
            preprocess_time = time.time() - preprocess_start

            # NPU推理
            inference_start = time.time()
            outputs = self.session.infer([blob])  # 同步推理
            inference_time = time.time() - inference_start

            # 后处理
            postprocess_start = time.time()
            dets = self.postprocess(outputs, {
                "padding": padding_info,
                "original_shape": orig_shape
            })
            postprocess_time = time.time() - postprocess_start

            # 处理总时间
            total_frame_time = time.time() - frame_start

            # 绘制检测框并输出
            render_start = time.time()
            for x1, y1, x2, y2, conf, cls_id in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_id}:{conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)
            render_time = time.time() - render_start

            out.write(frame)

            # 累计性能统计
            processing_times.append({
                "preprocess": preprocess_time * 1000,  # 毫秒
                "inference": inference_time * 1000,
                "postprocess": postprocess_time * 1000,
                "render": render_time * 1000,
                "total": total_frame_time * 1000
            })

            # 更新总统计
            self.total_inference_time += inference_time
            self.total_frames = frame_counter

            # 显示处理进度
            if show_progress:
                if total_frames > 0:
                    progress = frame_counter / total_frames * 100
                    print(f"\r🔄 处理进度: {frame_counter}/{total_frames} ({progress:.1f}%) | "
                          f"推理: {inference_time * 1000:.1f}ms | "
                          f"总帧时: {total_frame_time * 1000:.1f}ms", end="")
                else:
                    print(f"\r🔄 处理帧数: {frame_counter} | "
                          f"推理: {inference_time * 1000:.1f}ms | "
                          f"总帧时: {total_frame_time * 1000:.1f}ms", end="")

        # 处理完成
        total_elapsed = time.time() - start_time

        # 性能摘要
        print("\n\n✅ 处理完成!")
        print(f"  - 处理总时间: {total_elapsed:.2f}秒")
        print(f"  - 平均帧率: {frame_counter / total_elapsed:.1f}FPS")

        # 各阶段耗时分析
        if processing_times:
            avg_prep = sum([t['preprocess'] for t in processing_times]) / frame_counter
            avg_inf = sum([t['inference'] for t in processing_times]) / frame_counter
            avg_post = sum([t['postprocess'] for t in processing_times]) / frame_counter
            avg_render = sum([t['render'] for t in processing_times]) / frame_counter
            avg_total = sum([t['total'] for t in processing_times]) / frame_counter

            print("\n📊 性能分析 (平均值):")
            print(f"  - 预处理: {avg_prep:.1f}ms ({avg_prep / avg_total * 100:.1f}%)")
            print(f"  - NPU推理: {avg_inf:.1f}ms ({avg_inf / avg_total * 100:.1f}%)")
            print(f"  - 后处理: {avg_post:.1f}ms ({avg_post / avg_total * 100:.1f}%)")
            print(f"  - 渲染: {avg_render:.1f}ms ({avg_render / avg_total * 100:.1f}%)")
            print(f"  - 单帧总耗时: {avg_total:.1f}ms")

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 配置路径
    model_path = "./runs/train/train/weights/best.om"
    video_path = "./datasets/video/dog.mp4"
    output_video = "./output_dog.mp4"

    # 初始化并运行
    detector = YOLOv11_NPU_Inference(model_path)

    # 添加超时时间打印（防止长时间无响应）
    print("⏱️ 开始执行...")
    start_time = time.time()

    try:
        detector.run_video(video_path, output_path=output_video)
    finally:
        # 最终状态报告
        total_time = time.time() - start_time
        print(f"\n🔥 NPU利用率统计:")
        print(f"  - 总推理时间: {detector.total_inference_time:.2f}秒")
        print(f"  - 总处理帧数: {detector.total_frames}帧")
        print(f"  - 平均推理速度: {detector.total_inference_time / detector.total_frames * 1000:.1f}ms/帧")

        if detector.total_frames > 0:
            fps = detector.total_frames / total_time
            print(f"  - 端到端FPS: {fps:.1f} (含视频编解码)")

        print(f"\n💾 输出视频已保存至: {output_video}")