from models import *
from src.utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


from PIL import Image


# load weights and set defaults
config_path='models/yolov3/config/yolov3.cfg'
weights_path='models/yolov3/config/yolov3.weights'
class_path='models/yolov3/config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

class  Peoplenet():
    def __init__(self):
        # load model and put into eval mode
        model = Darknet(config_path, img_size=img_size)
        model.load_weights(weights_path)
        model.cuda()
        self.model= model.eval()

        self.classes = utils.load_classes(class_path)
        self.Tensor = torch.cuda.FloatTensor

    

    def detect_image(frame):
        img = Image.fromarray(frame)
        # scale and pad image
        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)),
            transforms.ToTensor(),
            ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
        return detections[0]