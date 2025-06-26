import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


def infer_api_static():
    device_id = 0
    model_path = "./runs/train/train/weights/best.om"
    image_path = './datasets/cat_dog_split/images/train/cat_72.jpg'

    # 1. 准备输入数据
    image = cv2.imread(image_path)
    session = InferSession(device_id, model_path)
    input_shape = session.get_inputs()[0].shape  # 获取模型输入维度 [N,C,H,W]
    _, _, h, w = input_shape

    # 预处理（调整尺寸+转NCHW）
    resized = cv2.resize(image, (w, h))
    input_data = resized.transpose(2, 0, 1)[None].astype(np.float32)  # HWC -> CHW -> NCHW

    # 2. 执行推理
    outputs = session.infer([input_data], mode='static')
    print("原始输出形状:", outputs[0].shape)  # 关键：打印实际维度

    # 3. 动态处理输出维度
    pred = outputs[0].squeeze(0)  # 去除batch维度

    # 情况1：输出为展平的预测结果 [8400, 6]
    if pred.ndim == 2 and pred.shape[1] == 6:
        print("检测到展平格式输出，无需转置")
        # 此处添加后处理（如NMS）

    # 情况2：输出为多维张量 [3, 40, 40, 6]
    elif pred.ndim == 4:
        print("检测到多维输出，执行维度重组")
        # 重组为 [预测数量, 6]
        pred = pred.transpose(0, 2, 3, 1).reshape(-1, 6)

    # 情况3：其他维度
    else:
        print(f"警告：非常规输出维度 {pred.shape}")
        # 尝试通用处理
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)  # CHW -> HWC

    # 4. 转换为图像兼容格式
    if pred.ndim == 3:  # 确认是HWC格式
        output_img = pred.astype(np.uint8)
        cv2.imwrite('./img/output.png', output_img)

    # 5. 资源释放
    if hasattr(session, 'destroy'):
        session.destroy()
    elif hasattr(session, 'free_resource'):  # 兼容旧版本
        session.free_resource()


infer_api_static()