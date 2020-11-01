import jetson.inference
import jetson.utils
import numpy as np
import cv2
import torch

from core.detection.face_recognizer import FaceRecognizer


class Detect:
    def __init__(self):

        self.face_detector = FaceRecognizer()

        self.people_net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
        self.facenet = jetson.inference.detectNet("facenet", threshold=0.2)

    def preProcess(self, color_image):

        bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
        # convert from BGR -> RGB
        rgb_img = jetson.utils.cudaAllocMapped(
            width=bgr_img.width, height=bgr_img.height, format="rgb8"
        )

        jetson.utils.cudaConvertColor(bgr_img, rgb_img)

        return rgb_img

    def detectPeople(self, color_image):

        # 600x1024 res ssd
        rgb_img = self.preProcess(color_image)

        detections = self.people_net.Detect(rgb_img)

        bboxes = self.getBBox(detections)

        return bboxes

    def getBBox(self, detections):
        predBox = []

        for detection in detections:
            bbox = (
                int(detection.Left),
                int(detection.Top),
                int(detection.Right),
                int(detection.Bottom),
            )

            area = (int(detection.Height), int(detection.Width))

            centroid = detection.Center

            predBox.append((bbox, area, centroid))

        # arr = np.array(predBox[0])
        # results = non_max_suppression(arr, probs=None, overlapThresh=0.65)

        return predBox

    def detectFaces(self, peoples, color_image):
        face_detections = []

        sladdedImage = color_image

        for people, area, _ in peoples:
            (h, w) = area
            (sx, sy, ex, ey) = people
            # half height of bbox
            half = int(h / 2)
            cropped = color_image[sy : sy + half, sx:ex]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            testimg = self.preProcess(cropped)
            # face1 = self.face_detector.predict_faces(testimg)
            # print(face1)
            face = self.facenet.Detect(testimg)
            bbox = self.getBBox(face)

            sladdedImage = self.sladFaces(bbox, color_image, people)

        return sladdedImage

    def sladFaces(self, bbox, color_image, people):
        (sx, sy, ex, ey) = people
        for roi, area, _ in bbox:
            (h, w) = area

            (dsx, dsy, dex, dey) = roi

            sx = dsx + sx
            sy = dsy + sy
            ex = dex + ex
            ey = dey + ey
            face = color_image[sy : (sy + h), sx : (sx + w)]

            face = self.sladFace(face)

            color_image[sy : (sy + h), sx : (sx + w)] = face

        return color_image

    def sladFace(self, image, factor=3.0):

        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)
        # ensure the width of the kernel is odd
        if kW % 2 == 0:
            kW -= 1
        # ensure the height of the kernel is odd
        if kH % 2 == 0:
            kH -= 1
        # apply a Gaussian blur to the input image using our computed
        # kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)