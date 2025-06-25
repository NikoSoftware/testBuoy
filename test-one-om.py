import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


def infer_api_static():
    device_id = 0
    model_path = "./runs/train/train/weights/best.om"
    image_path = 'datasets/cat_dog_split/images/train/cat_72.jpg'
    image = cv2.imread(image_path)

    # create session of om model for inference
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    shape0 = session.get_inputs()[0].shape
    print(shape0)
    height, width = shape0[2], shape0[3]
    resized_image = cv2.resize(image, (width, height))
    image_array = np.array(resized_image).astype(np.float32)

    feeds = [image_array]

    # execute inference, inputs is ndarray list and outputs is ndarray list
    outputs = session.infer(feeds, mode='static')
    print(outputs[0].shape)

    np.set_printoptions(threshold=np.inf)
    print(outputs[0])

    cv2.imwrite('./img/output.png', outputs[0])


    # free model resource and device context of session
    session.free_resource()


infer_api_static()
