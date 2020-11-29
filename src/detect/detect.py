import jetson.inference
import jetson.utils
import numpy as np
import cv2
import torch
from imutils.object_detection import non_max_suppression
from ..detection.face_recognizer import FaceRecognizer
import os
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from ..detection.maskDetection import MaskDetector
class Detect:
    def __init__(self):
          
        
        self.masknet = MaskDetector()
        self.masknet.load_state_dict(torch.load("models/face_mask.ckpt")['state_dict'], strict=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.masknet = self.masknet.to(self.device)
        self.masknet.eval()  
        
        self.transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])      
        
        self.labelColor = [(10, 0, 255), (10, 255, 0)]

        self.face_detector = FaceRecognizer()
        self.people_net = jetson.inference.detectNet("pednet", threshold=0.6)
        #self.facenet = jetson.inference.detectNet("facenet", threshold=0.2)

 

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


    def detect_dnn(self,frame):


        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):

            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                # face = img_to_array(face)
                # face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        print(faces)


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
        sladdedImage = self.sladFace_facenet(bboxes,color_image)


    def sladFace_facenet(self,faceboxes,color_image):

        for (roi,_,_) in faceboxes:
            (dsx, dsy, dex, dey) = roi
            h = dey - dsy
            w = dex - dsx
            sx = dsx 
            sy = dsy 
            ex = dex
            ey = dey 
            face = color_image[int(sy) : int((sy + h)), int(sx) : int((sx + w))]

            face = self.sladFace(face)

            color_image[int(sy) : int((sy + h)), int(sx) : int((sx + w))] = face

        return color_image




    def detect_face_mask(self, face_crops, face_boxes, color_image):
        if len(face_crops) > 0:
            if len(face_crops[0])>0:


                faces = []

                for face,box in zip(face_crops,face_boxes):

                    if not face.size ==0:
                        (startX, startY, endX, endY) = box
                        height = endY-startY
                        width = endX-startX

                         
                        output = self.masknet(self.transformations(face).unsqueeze(0).to(self.device))
                        _, predicted = torch.max(output.data, 1)


                        cv2.rectangle(color_image,
                            (startX, startY),
                            (startX + width, startY + height),
                            self.labelColor[predicted],
                            thickness=2)
                       

              

                       
                    

    def detectFaces(self, color_image):
        faceCrops = []
        faces_boxes = self.face_detector.predict_faces(color_image)
        if len(faces_boxes) > 0:

            for facebox in faces_boxes:
                (dsx, dsy, dex, dey) = facebox
                crop = color_image[int(dsy) : int((dey)), int(dsx) : int((dex))]
                faceCrops.append(crop)
                #face = self.sladFace(crop)
                #color_image[int(dsy) : int((dey)), int(dsx) : int((dex))] = face

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

    def sladFace(self,image, factor=3.0):

        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        try:
            (h, w) = image.shape[:2]
            
            kW = int(w / factor)
            kH = int(h / factor)
            # ensure the width of the kernel is odd
            if kW == 0:
                kW =1
            elif kW % 2 == 0:
                kW -= 1
            # ensure the height of the kernel is odd
            if kH ==0:
                kH =1
            elif kH % 2 == 0:
                kH -= 1
            # apply a Gaussian blur to the input image using our computed
            # kernel size
        except Exception as e:
           print(str(e))

        else:

            return cv2.GaussianBlur(image, (kW, kH), 0)