import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import signal
import sys

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
VIDEO_PATH = "./datasets/video/cat_blibli_7.mp4"
CLASS_NAMES = ["cat", "dog"]  # 类别名称
CONF_THRESH = 0.6  # 置信度阈值
NMS_THRESH = 0.65  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = False  # 是否显示实时检测窗口


# ================ 关键修改1：添加输出维度验证 ================
def validate_output_shape(outputs):
    """验证模型输出维度是否符合预期"""
    output_shape = outputs[0].shape
    print(f"模型输出维度: {output_shape}")
    if output_shape[1] != 6:
        print(f"⚠️ 警告: 输出通道数应为6 (xywh+obj+2类), 实际为{output_shape[1]}")
    return output_shape


# ================ 关键修改2：优化预处理归一化 ================
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0

    # 添加归一化统计信息
    print(f"预处理统计: min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}")

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img


# ================ 关键修改3：重构后处理逻辑 ================
def postprocess(outputs, orig_shape, input_size=(640, 640)):
    """后处理（适配[1,6,8400]输出结构）"""
    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = input_size

    # 验证输出维度
    output_shape = validate_output_shape(outputs)
    predictions = np.squeeze(outputs[0])  # [6,8400]

    # 添加输出值统计
    print(f"输出值统计: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")

    predictions = predictions.transpose((1, 0))  # [8400,6]

    # ================= 修正解析逻辑 =================
    boxes = predictions[:, :4].copy()  # 前4列为边界框 (cx, cy, w, h)
    objectness = predictions[:, 4].copy()  # 第5列为目标存在置信度
    class_logits = predictions[:, 5].copy()  # 第6列为类别分数（二分类逻辑回归值）

    # ================= 计算类别概率 =================
    # 使用sigmoid将类别logit转换为概率
    dog_probs = 1 / (1 + np.exp(-class_logits))  # 狗的概率
    cat_probs = 1 - dog_probs  # 猫的概率
    class_scores = np.column_stack((cat_probs, dog_probs))

    # ================= 计算最终置信度 =================
    # 置信度 = 目标存在概率 * 最大类别概率
    max_class_scores = np.max(class_scores, axis=1)
    confidences = objectness * max_class_scores
    class_ids = np.argmax(class_scores, axis=1)

    # ================= 调试输出 =================
    print("\n前5个检测框详细统计:")
    for i in range(5):
        print(
            f"框{i}: obj={objectness[i]:.3f}, cat={cat_probs[i]:.3f}, dog={dog_probs[i]:.3f}, 置信度={confidences[i]:.3f}")

    # 边界框格式转换 (cx, cy, w, h) -> (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.column_stack((x1, y1, x2, y2))

    # 坐标缩放（原始图像尺寸）
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # ================= 过滤低置信度检测框 =================
    valid_mask = confidences > CONF_THRESH
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]

    # NMS处理
    detections = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confidences.tolist(),
            score_threshold=CONF_THRESH,
            nms_threshold=NMS_THRESH
        )

        if indices is not None:
            if isinstance(indices, np.ndarray):
                if indices.ndim == 2:  # OpenCV 4.5.3+兼容
                    indices = indices[:, 0]
                indices = indices.astype(int)

            for idx in indices:
                class_id = class_ids[idx]
                confidence = confidences[idx]

                x1, y1, x2, y2 = boxes[idx]
                x1 = max(0, min(orig_w - 1, x1))
                y1 = max(0, min(orig_h - 1, y1))
                x2 = max(0, min(orig_w - 1, x2))
                y2 = max(0, min(orig_h - 1, y2))

                detections.append({
                    "class": CLASS_NAMES[class_id],
                    "confidence": float(confidence),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })

    print(f"有效检测框: {len(detections)} (过滤前: {len(valid_mask)})")
    return detections


def main():
    session = InferSession(device_id=0, model_path=MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    def signal_handler(sig, frame):
        print("\n捕获到Ctrl+Z信号，正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGTSTP, signal_handler)

    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return

    print(f"视频分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    frame_count = 0
    total_preprocess = 0.0
    total_inference = 0.0
    total_postprocess = 0.0
    total_frame = 0.0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        frame_count += 1

        # 预处理
        preprocess_start = time.time()
        orig_h, orig_w = frame.shape[:2]
        blob = preprocess(frame)
        preprocess_time = time.time() - preprocess_start

        # 推理
        inference_start = time.time()
        outputs = session.infer([blob])
        inference_time = time.time() - inference_start

        # 后处理
        postprocess_start = time.time()
        detections = postprocess(outputs, (orig_h, orig_w))
        postprocess_time = time.time() - postprocess_start

        frame_time = time.time() - frame_start

        # 累计时间
        total_preprocess += preprocess_time
        total_inference += inference_time
        total_postprocess += postprocess_time
        total_frame += frame_time

        # 打印信息
        print(f"\n帧 {frame_count} 时间统计:")
        print(f"预处理: {preprocess_time * 1000:.2f}ms | "
              f"推理: {inference_time * 1000:.2f}ms | "
              f"后处理: {postprocess_time * 1000:.2f}ms | "
              f"总帧耗时: {frame_time * 1000:.2f}ms")

        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i + 1}: {det['class']} | "
                  f"置信度: {det['confidence']:.4f} | "
                  f"位置: {det['box']}")

            # 绘制结果
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if SHOW_WINDOW:
            cv2.imshow('Detection', frame)
            cv2.resizeWindow('Detection', 640, 640)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.001)

    # 性能汇总
    avg_fps = frame_count / total_frame
    print("\n===== 性能汇总 =====")
    print(f"总帧数: {frame_count} | 平均FPS: {avg_fps:.2f}")
    print(f"平均预处理时间: {total_preprocess / frame_count * 1000:.2f}ms")
    print(f"平均推理时间: {total_inference / frame_count * 1000:.2f}ms")
    print(f"平均后处理时间: {total_postprocess / frame_count * 1000:.2f}ms")

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()