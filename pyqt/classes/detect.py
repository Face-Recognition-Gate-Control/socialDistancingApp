import jetson.inference
import jetson.utils
import numpy as np
import cv2
import torch
from imutils.object_detection import non_max_suppression
from core.detection.face_recognizer import FaceRecognizer
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


class Detect:
    def __init__(self):

        self.face_detector = FaceRecognizer()
        self.maskNet = load_model("pyqt/classes/mask_detector.model")

        self.people_net = jetson.inference.detectNet("pednet", threshold=0.8)
        # self.facenet = jetson.inference.detectNet("facenet", threshold=0.2)

    def preProcess(self, color_image):

        bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=False)
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

        # bboxes = [lis[0] for lis in predBox]
        # arr = np.array(bboxes)

        # test = non_max_suppression(arr, probs=None, overlapThresh=0.3)

        # newpredbox = []
        # for i, (bbox, area, centroid) in enumerate(predBox):
        #     bbox = tuple(test[i])

        #     newpredbox.append((bbox, area, centroid))

        return predBox

    def detectFaces2(self, peoples, color_image):
        face_detections = []

        sladdedImage = color_image

        for person, area, _ in peoples:
            (h, w) = area
            (sx, sy, ex, ey) = person
            # half height of bbox
            half = int(h / 2)
            cropped = color_image[sy : sy + half, sx:ex]

            testimg = self.preProcess(cropped)
            # faceBox = self.face_detector.predict_faces(cropped)
            # sladdedImage = self.sladFaces2(faceBox, color_image, person)

            # face = self.facenet.Detect(testimg)
            # bbox = self.getBBox(face)

            # sladdedImage = self.sladFaces(bbox, color_image, person)
            sladdedImage = self.anonymizeFace_general(color_image, person, area)

        return sladdedImage

    def decetesFaces_facenet(self, color_image):
        img = self.preProcess(color_image)
        faceboxes = self.facenet.Detect(img)
        bboxes = self.getBBox(faceboxes)
        sladdedImage = self.sladface

    def detect_face_mask(self, face_crops, face_boxes, color_image):
        if len(face_crops) > 0:

            faces = []

            for face in face_crops:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)

            masks = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(masks, batch_size=32)

            for pred, face in zip(preds, face_boxes):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(color_image, (startX, startY), (endX, endY), color, 2)

    def detectFaces(self, color_image):
        faceCrops = []
        faces_boxes = self.face_detector.predict_faces(color_image)
        if len(faces_boxes) > 0:

            for facebox in faces_boxes:
                (dsx, dsy, dex, dey) = facebox
                crop = color_image[int(dsy) : int((dey)), int(dsx) : int((dex))]
                faceCrops.append(crop)
                face = self.sladFace(crop)
                color_image[int(dsy) : int((dey)), int(dsx) : int((dex))] = face

        return (faceCrops, faces_boxes)

    def sladFacesRbf(self, faceBoxes, color_image, personBox):
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

    def anonymizeFace_general(self, color_image, personBox, area):
        (sx, sy, ex, ey) = personBox
        (h, w) = area

        face = color_image[int(sy) : int(w) / 4, int(sx) : int(sx + w)]
        face = self.sladFace(face)
        color_image[int(sy) : int(w) / 4, int(sx) : int(sx + w)] = face

        return color_image

    def sladFaces(self, faceBox, color_image, personBox):
        (sx, sy, ex, ey) = personBox
        for roi, area, _ in faceBox:
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

    def sladFace(self, image, factor=2.0):

        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        try:
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
        except Exception as e:
            print(e)

        else:

            return cv2.GaussianBlur(image, (kW, kH), 0)