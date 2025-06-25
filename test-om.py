import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import signal
import sys
import os

# ====================== FP16优化配置 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
VIDEO_PATH = "./datasets/test/30386095338-1-192.mp4"
CLASS_NAMES = ["cat", "dog"]  # 类别名称
CONF_THRESH = 0.55  # 降低置信度阈值提高召回率
NMS_THRESH = 0.6  # 调整NMS阈值平衡精度与召回
INPUT_SIZE = (640, 640)
SHOW_WINDOW = False
CALIBRATION_ENABLED = False  # 启用量化校准模式
CALIBRATION_FILE = "./calibration_data.npy"  # 校准数据保存路径
CALIBRATION_SAMPLES = 100  # 校准样本数量
USE_FP32_MODE = False  # 关键修改：启用FP16模式[6](@ref)

# ====================== 量化校准数据收集 (FP16适配版) ======================
if CALIBRATION_ENABLED and not os.path.exists(CALIBRATION_FILE):
    print("[FP16优化] 正在收集量化校准数据...")
    calibration_data = []
    cap_cal = cv2.VideoCapture(0)
    cap_cal.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_cal.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(CALIBRATION_SAMPLES):
        ret, frame = cap_cal.read()
        if not ret:
            continue

        # 旋转裁剪
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame.shape[:2]
        start_x, start_y = (w - 640) // 2, (h - 640) // 2
        frame = frame[start_y:start_y + 640, start_x:start_x + 640]

        # FP16适配预处理 - 确保数值范围合理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0.0, 1.0)  # 限制数值范围防止FP16溢出
        img = np.transpose(img, (2, 0, 1))
        calibration_data.append(img)

    np.save(CALIBRATION_FILE, np.array(calibration_data))
    cap_cal.release()
    print(f"[FP16优化] 已保存 {CALIBRATION_SAMPLES} 个校准样本")


# ====================== 预处理优化 (FP16适配版) ======================
def preprocess(frame, calibration=False):
    """FP16优化版预处理：确保数值范围适配FP16精度"""
    # 旋转裁剪
    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = rotated.shape[:2]
    start_x, start_y = (w - 640) // 2, (h - 640) // 2
    frame = rotated[start_y:start_y + 640, start_x:start_x + 640]

    # 自适应直方图均衡化（CLAHE）
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 转换颜色空间和归一化
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # FP16适配 - 确保数值范围合理
    img = np.clip(img, 0.0, 1.0)

    # 数据增强校准
    if calibration:
        img = img * 0.9 + 0.05  # 模拟训练时的随机亮度变化
        img = np.clip(img, 0.0, 1.0)  # 再次限制范围

    # 格式转换
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度

    return img


# ====================== DIoU-NMS后处理优化 ======================
def diou_nms(boxes, scores, iou_threshold):
    """DIoU-NMS实现：考虑中心点距离的改进NMS算法"""
    if len(boxes) == 0:
        return []

    # 按置信度排序
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 计算DIoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # 计算交集区域
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # 计算并集区域
        area_i = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1)
        union = area_i + area_j - intersection

        # 计算IoU
        iou = intersection / union

        # 计算中心点距离
        c_x_i = (boxes[i, 0] + boxes[i, 2]) / 2
        c_y_i = (boxes[i, 1] + boxes[i, 3]) / 2
        c_x_j = (boxes[order[1:], 0] + boxes[order[1:], 2]) / 2
        c_y_j = (boxes[order[1:], 1] + boxes[order[1:], 3]) / 2

        # 计算最小包围框对角线距离
        c_w = np.maximum(boxes[i, 2], boxes[order[1:], 2]) - np.minimum(boxes[i, 0], boxes[order[1:], 0])
        c_h = np.maximum(boxes[i, 3], boxes[order[1:], 3]) - np.minimum(boxes[i, 1], boxes[order[1:], 1])
        c_diag = c_w ** 2 + c_h ** 2

        # 计算中心点距离
        center_dist = (c_x_i - c_x_j) ** 2 + (c_y_i - c_y_j) ** 2

        # 计算DIoU
        diou = iou - center_dist / c_diag

        # DIoU阈值筛选
        idx = np.where(diou <= iou_threshold)[0]
        order = order[idx + 1]

    return keep


# ====================== 后处理优化 (FP16适配版) ======================
def postprocess(outputs, orig_shape, input_size=(640, 640)):
    """FP16优化版后处理：处理FP16输出并确保数值稳定性"""
    # 确保输出为FP32处理
    if outputs[0].dtype == np.float16:
        outputs = [output.astype(np.float32) for output in outputs]

    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = input_size

    # 处理模型输出
    predictions = np.squeeze(outputs[0])

    # 自适应输出维度处理
    if predictions.ndim == 2:
        predictions = predictions.transpose((1, 0))
    elif predictions.ndim == 3:
        predictions = predictions.reshape(predictions.shape[1], -1).transpose((1, 0))

    # 分离边界框和类别分数
    boxes = predictions[:, :4].copy()
    scores = predictions[:, 4:4 + len(CLASS_NAMES)]  # 仅取实际类别数

    # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2

    # 缩放回原始图像尺寸
    scale_x, scale_y = orig_w / model_w, orig_h / model_h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # 应用DIoU-NMS
    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    # 筛选高置信度检测
    mask = confidences > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # DIoU-NMS处理
    if len(boxes) > 0:
        indices = diou_nms(boxes, confidences, NMS_THRESH)
    else:
        indices = []

    # 提取有效检测结果
    detections = []
    for idx in indices:
        class_id = class_ids[idx]
        confidence = confidences[idx]

        # 获取缩放后的边界框坐标
        x1, y1, x2, y2 = boxes[idx]

        # 确保坐标在图像范围内
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        detections.append({
            "class": CLASS_NAMES[class_id],
            "confidence": float(confidence),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return detections


# ====================== 模型初始化优化 (FP16适配版) ======================
def init_model():
    """模型初始化：适配FP16模式"""
    device_id = 0
    print(f"[FP16优化] 正在初始化模型 (FP16模式: {not USE_FP32_MODE})")

    # 量化校准处理
    if CALIBRATION_ENABLED and os.path.exists(CALIBRATION_FILE):
        print("[FP16优化] 应用量化校准数据")
        calibration_data = np.load(CALIBRATION_FILE)

        # 创建校准会话
        cal_session = InferSession(device_id, MODEL_PATH)

        # 运行校准推理 (使用FP16模式)
        for data in calibration_data:
            cal_session.infer([data], mode=None if USE_FP32_MODE else "fp16")

    # 创建正式推理会话
    session = InferSession(device_id, MODEL_PATH)


    return session


# ====================== 主函数优化 ======================
def main():
    # 初始化模型
    session = init_model()

    # 打开视频
    cap = cv2.VideoCapture(0)

    # 注册信号处理器
    def signal_handler(sig, frame):
        print("\n捕获到Ctrl+Z信号，正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGTSTP, signal_handler)

    if not cap.isOpened():
        print("错误: 无法打开USB摄像头")
        return

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"摄像头分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    # 性能统计
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

        # ====================== 帧处理开始 ======================
        frame_start = time.time()
        frame_count += 1

        # 1. 预处理
        preprocess_start = time.time()
        blob = preprocess(frame)
        preprocess_time = time.time() - preprocess_start

        # 2. 推理 (使用FP16模式)
        inference_start = time.time()
        outputs = session.infer([blob], mode="fp32" if USE_FP32_MODE else None)
        inference_time = time.time() - inference_start

        # 3. 后处理
        postprocess_start = time.time()
        detections = postprocess(outputs, (640, 640))  # 裁剪后尺寸固定为640x640
        postprocess_time = time.time() - postprocess_start

        # 4. 帧总处理时间
        frame_time = time.time() - frame_start

        # 累计时间统计
        total_preprocess += preprocess_time
        total_inference += inference_time
        total_postprocess += postprocess_time
        total_frame += frame_time

        # ====================== 性能监控 ======================
        print(f"\n帧 {frame_count} 时间统计:")
        print(f"预处理: {preprocess_time * 1000:.2f}ms | "
              f"推理: {inference_time * 1000:.2f}ms | "
              f"后处理: {postprocess_time * 1000:.2f}ms | "
              f"总帧耗时: {frame_time * 1000:.2f}ms")

        # 打印检测结果
        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i + 1}: {det['class']} | "
                  f"置信度: {det['confidence']:.4f} | "
                  f"位置: [{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]")

            # 在图像上绘制结果
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # ====================== 显示控制 ======================
        if SHOW_WINDOW:
            # 显示带FPS的实时结果
            fps_text = f"FPS: {1.0 / frame_time:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('YOLOv11 Detection (FP16优化版)', frame)
            cv2.resizeWindow('YOLOv11 Detection (FP16优化版)', 640, 640)

            # 检查退出按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ====================== 最终性能统计 ======================
    end_time = time.time()
    total_duration = end_time - start_time
    fps = frame_count / total_duration

    print("\n" + "=" * 50)
    print("最终性能统计:")
    print(f"总帧数: {frame_count} | 总耗时: {total_duration:.2f}s | 平均FPS: {fps:.2f}")
    print(f"平均预处理时间: {total_preprocess / frame_count * 1000:.2f}ms/帧")
    print(f"平均推理时间: {total_inference / frame_count * 1000:.2f}ms/帧")
    print(f"平均后处理时间: {total_postprocess / frame_count * 1000:.2f}ms/帧")
    print(f"平均帧处理时间: {total_frame / frame_count * 1000:.2f}ms/帧")
    print("=" * 50)

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
