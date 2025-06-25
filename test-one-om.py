import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


def infer_api_static():
    device_id = 0
    model_path = "./runs/train/train/weights/best.om"
    image_path = 'datasets/cat_dog_split/images/train/cat_72.jpg'
    image = cv2.imread(image_path)

    session = InferSession(device_id, model_path)
    shape0 = session.get_inputs()[0].shape
    height, width = shape0[2], shape0[3]
    resized_image = cv2.resize(image, (width, height))
    image_array = np.array(resized_image).astype(np.float32)
    feeds = [image_array]

    outputs = session.infer(feeds, mode='static')

    # 处理输出数据
    output = outputs[0].squeeze(0)  # 去除 batch 维度
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    output = output.astype(np.uint8)  # 转换为 uint8

    cv2.imwrite('./img/output.png', output)

    # 正确释放资源
    session.destroy()  # 使用 destroy() 而非 free_resource()


infer_api_static()