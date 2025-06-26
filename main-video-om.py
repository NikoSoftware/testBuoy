import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import signal
import sys

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
VIDEO_PATH = "./datasets/video/cat_blibli_7.mp4"
CLASS_NAMES = ["cat", "dog"]  # 确保与训练时的类别顺序一致
CONF_THRESH = 0.6  # 置信度阈值
NMS_THRESH = 0.65  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = False  # 控制是否显示实时检测窗口


def preprocess(frame):
    """图像预处理 - 修复内存连续性问题"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    img = np.ascontiguousarray(img)  # 关键修复：确保内存连续[1](@ref)
    return img


def postprocess(outputs, orig_shape, input_size=(640, 640)):
    """后处理逻辑重构 - 适配模型输出[1,6,8400]格式"""
    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = input_size

    # 处理模型输出 [1,6,8400] -> [8400,6]
    predictions = np.squeeze(outputs[0])  # 移除batch维度 [6,8400]
    predictions = predictions.transpose((1, 0))  # 转置为[8400,6]

    # 分离边界框(4) + 目标置信度(1) + 类别分数(1)
    boxes = predictions[:, :4].copy()  # [x, y, w, h]
    obj_conf = predictions[:, 4]  # 目标置信度
    cls_scores = predictions[:, 5]  # 类别分数（二分类时为单值）

    # 计算最终类别置信度 = 目标置信度 * 类别分数
    confidences = obj_conf * cls_scores
    # 二分类处理：cls_scores>0.5为dog，否则为cat
    class_ids = (cls_scores > 0.5).astype(int)

    # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.column_stack([x1, y1, x2, y2])

    # 缩放回原始图像尺寸
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # ============== 关键修改：统一处理NMS返回值 ==============
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=NMS_THRESH
    )

    # 处理不同格式的返回值（元组/数组）
    if indices is not None:
        indices_np = np.array(indices)
        if indices_np.ndim == 2:  # 处理二维数组
            indices_np = indices_np[:, 0]
        indices_flat = indices_np.flatten().astype(int)
    else:
        indices_flat = np.array([], dtype=int)
    # ============== 修改结束 ==============

    detections = []
    for idx in indices_flat:
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

    return detections

def rotate_crop_frame(frame):
    """旋转裁剪为640x640 - 保持原逻辑"""
    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = rotated.shape[:2]
    start_x = (w - 640) // 2
    start_y = (h - 640) // 2
    return rotated[start_y:start_y + 640, start_x:start_x + 640]


def main():
    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)

    # 打开摄像头
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("错误: 无法打开USB摄像头")
        return

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"摄像头分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    # 注册信号处理器
    def signal_handler(sig, frame):
        print("\n释放资源中...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGTSTP, signal_handler)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 旋转裁剪为640x640
        # processed_frame = rotate_crop_frame(frame)
        processed_frame = frame
        print("输入尺寸:", processed_frame.shape)
        orig_h, orig_w = frame.shape[:2]

        # 预处理
        blob = preprocess(processed_frame)

        # 推理
        outputs = session.infer([blob])

        # 后处理
        detections = postprocess(outputs, (orig_h, orig_w))

        # 绘制检测结果
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示结果
        if SHOW_WINDOW:
            cv2.imshow('Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()