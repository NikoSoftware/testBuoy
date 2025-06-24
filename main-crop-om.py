import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time
import signal
import sys

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train/weights/best.om"
VIDEO_PATH = "./datasets/test/30386095338-1-192.mp4"
CLASS_NAMES = ["cat","dog"]  # 根据你的buoy.yaml修改类别名称
CONF_THRESH = 0.6  # 置信度阈值
NMS_THRESH = 0.65  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = True  # 控制是否显示实时检测窗口[2,3](@ref)





def preprocess(frame):
    """图像预处理"""
    # 转换颜色空间和调整尺寸
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, INPUT_SIZE)

    # 归一化并转换格式
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img


def postprocess(outputs, orig_shape, input_size=(640, 640)):
    """优化版后处理：解决坐标偏移问题
    Args:
        outputs: 模型原始输出 [1, 84, 8400]
        orig_shape: 原始图像尺寸 (h, w)
        input_size: 模型输入尺寸 (w, h)，默认640x640
    """
    # 解包尺寸
    orig_h, orig_w = orig_shape[:2]
    model_w, model_h = input_size

    # 处理模型输出
    predictions = np.squeeze(outputs[0])  # 移除batch维度 [84, 8400]
    predictions = predictions.transpose((1, 0))  # 转置为[8400, 84]

    # 分离边界框和类别分数
    boxes = predictions[:, :4].copy()  # 使用copy避免原始数据被修改
    scores = predictions[:, 4:]

    # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
    # 修复：先计算所有值再赋值，避免中间值覆盖问题
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # 更新boxes数组
    boxes[:, 0] = x1
    boxes[:, 1] = y1
    boxes[:, 2] = x2
    boxes[:, 3] = y2

    # 关键修复：缩放回原始图像尺寸
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    boxes[:, [0, 2]] *= scale_x  # 缩放x坐标
    boxes[:, [1, 3]] *= scale_y  # 缩放y坐标

    # 应用NMS
    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    # 兼容不同OpenCV版本的NMSBoxes返回值
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=NMS_THRESH
    )

    # 提取有效检测结果
    detections = []
    if indices is not None:
        # 统一索引格式处理
        if isinstance(indices, np.ndarray):
            if indices.ndim == 2:  # OpenCV 4.5.3+ 返回二维数组
                indices = indices[:, 0]
            indices = indices.astype(int)

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


def rotate_crop_frame(frame):

    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w = rotated.shape[:2]

    start_x = (w-640)//2

    start_y =(h-640)//2

    cropped = rotated[start_y:start_y+640, start_x:start_x+640]


    return cropped




def main():
    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)



    # 打开视频
    cap = cv2.VideoCapture(0)
    # 注册SIGTSTP信号处理器

    def signal_handler(sig, frame):
        print("\n捕获到Ctrl+Z信号，正在释放摄像头资源...")
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭窗口
        sys.exit(0)  # 优雅退出
    signal.signal(signal.SIGTSTP, signal_handler)

    if not cap.isOpened():
        print(f"错误: 无法打开usb摄像头失败")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    #打印摄像头 信息
    print(f"分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

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

        #旋转裁剪为640*640
        frame = rotate_crop_frame(frame)

        # ====================== 帧开始计时 ======================
        frame_start = time.time()
        frame_count += 1

        # 1. 预处理计时
        preprocess_start = time.time()
        orig_h, orig_w = frame.shape[:2]
        blob = preprocess(frame)
        preprocess_time = time.time() - preprocess_start

        # 2. 推理计时
        inference_start = time.time()
        outputs = session.infer([blob])
        inference_time = time.time() - inference_start

        # 3. 后处理计时
        postprocess_start = time.time()
        detections = postprocess(outputs, (orig_h, orig_w))
        postprocess_time = time.time() - postprocess_start

        # 4. 帧总处理时间
        frame_time = time.time() - frame_start

        # 累计时间统计
        total_preprocess += preprocess_time
        total_inference += inference_time
        total_postprocess += postprocess_time
        total_frame += frame_time

        # ====================== 打印时间信息 ======================
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

        # ====================== 显示控制开关 ======================
        if SHOW_WINDOW:
            # 显示实时结果
            cv2.imshow('Detection', frame)
            cv2.resizeWindow('Detection', 640, 640)
            # 检查退出按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 无头模式下，仅延时1ms保持处理节奏
            time.sleep(0.001)

    # ====================== 最终性能统计 ======================
    # end_time = time.time()
    # total_duration = end_time - start_time
    # fps = frame_count / total_duration
    #
    # print("\n" + "=" * 50)
    # print("最终性能统计:")
    # print(f"总帧数: {frame_count} | 总耗时: {total_duration:.2f}s | 平均FPS: {fps:.2f}")
    # print(f"平均预处理时间: {total_preprocess / frame_count * 1000:.2f}ms/帧")
    # print(f"平均推理时间: {total_inference / frame_count * 1000:.2f}ms/帧")
    # print(f"平均后处理时间: {total_postprocess / frame_count * 1000:.2f}ms/帧")
    # print(f"平均帧处理时间: {total_frame / frame_count * 1000:.2f}ms/帧")
    # print("=" * 50)

    # 清理资源
    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()