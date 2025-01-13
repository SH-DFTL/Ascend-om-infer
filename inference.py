import os
import cv2
import time
import numpy as np
from PIL import Image
from acl_net import ACL_Net
# from imgprocess import letterbox,postprocess,draw_detections
# from pt_infer import OM_Conf
# from ultralytics.models.yolo.detect import DetectionPredictor
import torch

# 全局变量
# ACL_MEM_MALLOC_HUGE_FIRST = 0
# ACL_MEMCPY_HOST_TO_DEVICE = 1
# ACL_MEMCPY_DEVICE_TO_HOST = 2

Classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
      
def transfer_pic(input_path):
    # 图像预处理
    input_path = os.path.abspath(input_path)
    with Image.open(input_path) as image_file:
        # 缩放为224*224
        img = image_file.resize((224, 224))
        # 转换为float32类型ndarray
        img = np.array(img).astype(np.float32)
    # 根据imageNet图片的均值和方差对图片像素进行归一化
    img -= [123.675, 116.28, 103.53]
    img /= [58.395, 57.12, 57.375]
    # RGB通道交换顺序为BGR
    img = img[:, :, ::-1]
    # resnet50为色彩通道在前
    img = img.transpose((2, 0, 1))
    # 返回并添加batch通道
    return np.array([img])


if __name__ == '__main__':

    time_ = time.time()
    image_path = './img/bus.jpg'
    img_ = cv2.imread(image_path)
    
    x_scale = img_.shape[1] / 640
    y_scale = img_.shape[0] / 640
    img = img_ / 255.
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    print(f'image preprocess time cost:{time.time() - time_}')
    
    model_path = './yolov11n.om'
    device_id = 0
    model = ACL_Net(model_path)
    start = time.time()
    result = model.forward([img])
    print(f'image inference time cost:{time.time() - start}') #shape:(705600,)
    #释放资源
    del model
    result = np.transpose(result[0].reshape(84,-1)) #shape:(705600,)->(84,8400)->(8400,84)

    print(result.shape)

    # np.save("./data/ominfer.npy",result)
    # postprocess
