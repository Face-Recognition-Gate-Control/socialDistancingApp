import jetson.inference
import jetson.utils
import numpy as np
import cv2
import torch
from imutils.object_detection import non_max_suppression

from core.detection.face_recognizer import FaceRecognizer


class Detect:
    def __init__(self):

        # self.face_detector = FaceRecognizer()

        self.people_net = jetson.inference.detectNet("ped-100", threshold=0.55)
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

        bboxes = [lis[0] for lis in predBox]
        arr = np.array(bboxes)

        test = non_max_suppression(arr, probs=None, overlapThresh=0.3)

        newpredbox = []
        for i, (bbox, area, centroid) in enumerate(predBox):
            bbox = tuple(test[i])

            newpredbox.append((bbox, area, centroid))

        return newpredbox

    def detectFaces(self, peoples, color_image):
        face_detections = []

        sladdedImage = color_image

        for person, area, _ in peoples:
            (h, w) = area
            (sx, sy, ex, ey) = person
            # half height of bbox
            half = int(h / 2)
            cropped = color_image[sy : sy + half, sx:ex]
            rezise = cv2.resize(cropped, (320, 320))
            cv2.imwrite("face.jpg", cropped)

            testimg = self.preProcess(rezise)
            # faceBox = self.face_detector.predict_faces(cropped)
            # sladdedImage = self.sladFaces2(faceBox, color_image, person)

            face = self.facenet.Detect(testimg)
            bbox = self.getBBox(face)

            sladdedImage = self.sladFaces(bbox, color_image, person)

        return sladdedImage

    def sladFaces2(self, faceBoxes, color_image, personBox):
        (sx, sy, ex, ey) = personBox
        faceBoxes = faceBoxes.tolist()
        for facebox in faceBoxes:

            (dsx, dsy, dex, dey) = facebox

            h = dey - dsy
            w = dex - dsx
            sx = dsx + sx
            sy = dsy + sy
            ex = dex + ex
            ey = dey + ey
            face = color_image[int(sy) : int((sy + h)), int(sx) : int((sx + w))]

            face = self.sladFace(face)

            color_image[int(sy) : int((sy + h)), int(sx) : int((sx + w))] = face

        return color_image

    def sladFaces(self, bbox, color_image, personBox):
        (sx, sy, ex, ey) = personBox
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