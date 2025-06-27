import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import signal
import sys

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
VIDEO_PATH = "./datasets/video/dog.mp4"
OUTPUT_PATH = "./output_dog.mp4"  # 输出视频路径
CLASS_NAMES = ["cat", "dog"]  # 确保与训练时的类别顺序一致
CONF_THRESH = 0.2  # 置信度阈值
NMS_THRESH = 0.2  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = False  # 控制是否显示实时检测窗口
FPS = 30  # 输出视频帧率


def preprocess(frame):
    """图像预处理 - 修复内存连续性问题"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    img = np.ascontiguousarray(img)  # 确保内存连续[5](@ref)
    return img


def postprocess(outputs, orig_shape, input_size=(640, 640)):
    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = input_size

    # 处理输出 [1,6,8400] -> [8400,6]
    predictions = np.squeeze(outputs[0])  # 移除batch维度
    predictions = predictions.transpose((1, 0))  # 转置为[8400,6]

    # 解析输出结构 (6维: cx,cy,w,h,conf,angle)
    boxes = predictions[:, :4].copy()  # 中心坐标+宽高
    confidences = predictions[:, 4]  # 置信度(单值)
    angles = predictions[:, 5]  # 旋转角度(弧度)

    # 转换边界框格式 (cx,cy,w,h) -> (x1,y1,x2,y2)
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

    # 应用NMS (注意: 单类别无需计算多类置信度)
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),  # 直接使用置信度
        score_threshold=CONF_THRESH,
        nms_threshold=NMS_THRESH
    )

    # 提取有效检测结果
    detections = []
    if indices is not None:
        # 统一索引格式
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().astype(int)

        for idx in indices:
            # 角度处理 (标准化到 -90°~90°)
            angle = angles[idx]
            if angle > np.pi / 2:
                angle -= np.pi  # 避免反向检测[7](@ref)

            # 单类别检测 (class_id固定为0或按需分配)
            class_id = 0  # 所有目标视为同一大类

            detections.append({
                "class": CLASS_NAMES[class_id],  # 输出类别名
                "confidence": float(confidences[idx]),
                "box": [
                    int(max(0, boxes[idx, 0])),
                    int(max(0, boxes[idx, 1])),
                    int(min(orig_w - 1, boxes[idx, 2])),
                    int(min(orig_h - 1, boxes[idx, 3]))
                ],
                "angle": np.degrees(angle)  # 弧度转角度
            })
    return detections


def main():
    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)
    print(f"已加载模型: {MODEL_PATH}")

    # 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps if fps > 0 else FPS,
                          (frame_width, frame_height))
    print(f"创建输出视频: {OUTPUT_PATH} ({frame_width}x{frame_height}, FPS: {fps if fps > 0 else FPS})")

    # 注册信号处理器
    def signal_handler(sig, frame):
        print("\n释放资源中...")
        cap.release()
        out.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGTSTP, signal_handler)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # 预处理
        blob = preprocess(frame)

        # NPU推理
        npu_start = time.time()
        outputs = session.infer([blob])
        npu_time = time.time() - npu_start

        # 后处理
        post_start = time.time()
        detections = postprocess(outputs, (orig_h, orig_w))
        post_time = time.time() - post_start

        # 绘制检测结果
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示处理信息
        info_text = f"Frame: {frame_count}/{total_frames} | NPU: {npu_time * 1000:.1f}ms | Post: {post_time * 1000:.1f}ms | Objs: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 写入输出视频
        out.write(frame)

        # 显示实时检测窗口
        if SHOW_WINDOW:
            cv2.imshow('YOLOv11 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 进度显示
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed
            print(f"已处理 {frame_count}/{total_frames} 帧, 平均FPS: {avg_fps:.1f}")

    # 计算最终性能
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {avg_fps:.1f}")
    print(f"输出视频已保存至: {OUTPUT_PATH}")

    # 释放资源
    cap.release()
    out.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()