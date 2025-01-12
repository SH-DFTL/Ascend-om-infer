import ultralytics
import numpy as np
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
import torch

Classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


class OM_Conf():
    def __init__(self):
        self.pt=False
        self.stride = 32
        self.fp16 = False
        self.overrides = {'task': 'detect', 'data': '/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml', 'imgsz': 640, 'single_cls': False, 'model': './models/yolo11s_bs1.om', 'conf': 0.25, 'batch': 1, 'save': True, 'mode': 'predict', 'save_txt': True}
        self.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def draw_detections(img, box, score, class_id):
    """
    作¨轾S佅¥佛¾佃~O䷾J纾X佈¶梾@派K佈°潚~D边潕~L栾F佒~L庠~G签⽀~B
    住~B录°﻾Z
        img: 潔¨乾N纾X佈¶梾@派K纾S彞~\潚~D轾S佅¥佛¾佃~O⽀~B
        box: 梾@派K佈°潚~D边潕~L栾F⽀~B
        score: 对幾T潚~D梾@派K佈~F录°⽀~B
        class_id: 梾@派K佈°潚~D潛®庠~G类佈« ID⽀~B

    达T佛~^﻾Z
        None
    """
    # 彏~P住~V边潕~L栾F潚~D佝~P庠~G
    x1, y1, w, h = box

    # 罎·住~V类佈«对幾T潚~D顾\罉²
    color = color_palette[class_id]

    # 作¨佛¾佃~O䷾J纾X佈¶边潕~L栾F
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # 佈~[建佌~E佐«类佈«佐~M佒~L佈~F录°潚~D庠~G签彖~G彜¬
    label = f"{Classes[class_id]}: {score:.2f}"
    print("label:",label, score, x1, y1, w, h)
    # 计签W庠~G签彖~G彜¬潚~D尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # 计签W庠~G签彖~G彜¬潚~D伾M置
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # 纾X佈¶填佅~E潚~D潟©形伾\为庠~G签彖~G彜¬潚~D罃~L彙¯
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                  cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__=='__main__':

    results = np.load('./ominfer.npy')

    #print(results.shape)
    overrides = {'task': 'detect', 'data': '/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml', 'imgsz': 640, 'single_cls': False, 'model': './models/yolo11s_bs1.om', 'conf': 0.25, 'batch': 1, 'save': True, 'mode': 'predict', 'save_txt': True}
    
    om_conf = OM_Conf()
    dp = DetectionPredictor(overrides=overrides)
    dp.model = om_conf
    dp.setup_source('./img')

    for dp.batch in dp.dataset:
        paths, im0s, s = dp.batch

        # Preprocess
        #im = dp.preprocess(im0s)


    #img0 = cv2.imread('./img/bus.jpg')
    #print(img0.shape)

        img = im0s[0] / 255.
        img = cv2.resize(img, (640, 640))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        #img0 = torch.from_numpy(img0).unsqueeze(0)
        results = torch.from_numpy(results).transpose(1,0).unsqueeze(0)
        print(results.shape)
        outputs = dp.postprocess(results,img,im0s)
        iuput = im0s[0]

        for i,output in enumerate(outputs):
            box = output.boxes[i]
            print(box)
            # score = output.scores[i]
            # class_id = output.class_ids[i]
            
            draw_detections(input, box, score, class_id)

            print("ii")


        # for i in outputs:
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_ids[i]
        #     draw_detections(input_image, box, score, class_id)
        # return input_image
        

        print(outputs)


   
    # draw_detections(input_image, box, score, class_id)