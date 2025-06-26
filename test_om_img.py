import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import os

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
IMAGE_PATH = "./img/test.jpg"  # 替换为您的测试图片路径
CLASS_NAMES = ["cat", "dog"]  # 类别名称
CONF_THRESH = 0.6  # 置信度阈值
NMS_THRESH = 0.65  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = True  # 是否显示检测窗口


def preprocess(frame):
    """图像预处理 - 修复内存连续性问题"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # 应用NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=NMS_THRESH
    )

    # 提取有效检测结果
    detections = []
    if indices is not None:
        indices = indices.flatten().astype(int)
        for idx in indices:
            class_id = class_ids[idx]
            confidence = confidences[idx]
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


def main():
    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)

    # 检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 图片文件不存在 - {IMAGE_PATH}")
        return

    # 读取图片
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"错误: 无法读取图片 - {IMAGE_PATH}")
        return

    # 旋转裁剪为640x640 (保持与原逻辑一致)
    frame = rotate_crop_frame(frame)
    orig_h, orig_w = frame.shape[:2]

    print(f"图片尺寸: {orig_w}x{orig_h}")
    print(f"模型输入尺寸: {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")

    # ====================== 处理开始计时 ======================
    start_time = time.time()

    # 1. 预处理
    preprocess_start = time.time()
    blob = preprocess(frame)
    preprocess_time = time.time() - preprocess_start

    # 2. 推理
    inference_start = time.time()
    outputs = session.infer([blob])
    inference_time = time.time() - inference_start

    # 3. 后处理
    postprocess_start = time.time()
    detections = postprocess(outputs, (orig_h, orig_w))
    postprocess_time = time.time() - postprocess_start

    # 总处理时间
    total_time = time.time() - start_time

    # ====================== 打印时间信息 ======================
    print("\n时间统计:")
    print(f"预处理: {preprocess_time * 1000:.2f}ms")
    print(f"推理: {inference_time * 1000:.2f}ms")
    print(f"后处理: {postprocess_time * 1000:.2f}ms")
    print(f"总耗时: {total_time * 1000:.2f}ms")

    # 打印检测结果
    print(f"\n检测到 {len(detections)} 个目标:")
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

    # ====================== 保存和显示结果 ======================
    result_path = "./img/result.jpg"
    cv2.imwrite(result_path, frame)
    print(f"\n结果已保存至: {result_path}")

    if SHOW_WINDOW:
        # 显示结果
        cv2.imshow('Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rotate_crop_frame(frame):
    # """旋转裁剪为640x640 - 保持原逻辑"""
    # rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # h, w = rotated.shape[:2]
    # start_x = (w - 640) // 2
    # start_y = (h - 640) // 2
    # return rotated[start_y:start_y + 640, start_x:start_x + 640]
    return cv2.resize(frame, (640, 640))


if __name__ == '__main__':
    main()