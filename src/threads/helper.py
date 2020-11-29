import cv2
from ..utils.post_process import *
from PyQt5.QtGui import QImage
from PyQt5.QtCore import (
   
    Qt)
import numpy as np


def rgbtoQimage( image):

    # https://stackoverflow.com/a/55468544/6622587
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    convertToQtFormat = QImage(
        rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
    )
    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

    return p

def alignImage( frames,align):
    
    frameset = align.process(frames)

    # # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()

    return aligned_depth_frame

def warning_complete():
    print("complete")

def getVectorsAndBbox( predictions, depth_frame):
    bboxes = []
    vectors = []
    test = []

    for bbox, area, centroid in predictions:

        (h, w) = area
        bboxes.append(bbox)
        x, y = centroid
        vectors.append(get3d(int(x), int(y), depth_frame))
        test.append((x, y, h))

    pred_bbox = (bboxes, vectors, test)

    return pred_bbox



def getColorImage(frames):
        color_Image =frames.get_color_frame()
        color_image = color_frame.get_data()
        color_image = np.asanyarray(color_image)
        return color_image