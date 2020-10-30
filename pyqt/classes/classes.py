from PyQt5.QtCore import (
    QObject,
    QRunnable,
    pyqtSlot,
    pyqtSignal,
    Qt,
    QThread,
    QThreadPool,
)
from PyQt5.QtGui import QImage
import cv2
import threading
from threading import Lock
import pyrealsense2 as rs
import os
from detect import config_caffe as config
import json
import numpy as np
from multiprocessing import Queue
from detect.detectCaffe import detect_people
from utils.post_process import *
import imutils
import simpleaudio as sa
import time
import jetson.inference
import jetson.utils


# initilize the queues for sharing recources between processes
original_frames = Queue(maxsize=0)
depthFrames = Queue(maxsize=0)
predicted_data = Queue(maxsize=0)
boundingBoxes = Queue(maxsize=0)
processed_frames = Queue(maxsize=0)
preProcessed_frames = Queue(maxsize=0)
detect_lock = Lock()
color_image2 = []
bbox_lock = Lock


class realsenseThread(QThread):
    def __init__(self, signals):
        super(realsenseThread, self).__init__()
        self.signals = signals
        self.signals.frameSelection.connect(self.updateSignal)
        self.signals.min_distance.connect(self.updateSignal2)
        self.selection = False
        self.minDistance = 1
        self.threadpool = QThreadPool()

    def updateSignal(self, value):
        self.selection = value

    def updateSignal2(self, value):

        self.minDistance = value

    @pyqtSlot()
    def run(self):

        self.people_net = jetson.inference.detectNet("pednet")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # load config file made
        # do adjustment in realsense depth quality tool
        jsonObj = json.load(open("configrealsense.json"))
        json_string = str(jsonObj).replace("'", '"')

        self.pipeline = rs.pipeline()
        rsconfig = rs.config()

        freq = int(jsonObj["stream-fps"])
        print("W: ", int(jsonObj["stream-width"]))
        print("H: ", int(jsonObj["stream-height"]))
        print("FPS: ", int(jsonObj["stream-fps"]))
        rsconfig.enable_stream(
            rs.stream.depth,
            int(jsonObj["stream-width"]),
            int(jsonObj["stream-height"]),
            rs.format.z16,
            int(jsonObj["stream-fps"]),
        )
        rsconfig.enable_stream(
            rs.stream.color,
            int(jsonObj["stream-width"]),
            int(jsonObj["stream-height"]),
            rs.format.bgr8,
            int(jsonObj["stream-fps"]),
        )
        camera = False

        while not camera:
            try:
                print("connecting to realsense")
                cfg = self.pipeline.start(rsconfig)
                dev = cfg.get_device()
                advnc_mode = rs.rs400_advanced_mode(dev)
                advnc_mode.load_json(json_string)

                # get depthscale from camera. converting distance to meter
                depth_scale = cfg.get_device().first_depth_sensor().get_depth_scale()

            except Exception as e:

                print(str(e))
                print("no device connected")

            finally:
                print("connected")
                camera = True
                self.startStreaming()

    def getBBox(self, detections):
        results = []

        for detection in detections:
            bbox = (
                int(detection.Left),
                int(detection.Top),
                int(detection.Right),
                int(detection.Bottom),
            )

            height = detection.Height

            centroid = detection.Center

            results.append((bbox, height, centroid))

        return results

    def preProcess(self, color_image):

        bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
        # convert from BGR -> RGB
        rgb_img = jetson.utils.cudaAllocMapped(
            width=bgr_img.width, height=bgr_img.height, format="rgb8"
        )

        jetson.utils.cudaConvertColor(bgr_img, rgb_img)

        return rgb_img

    def rgbtoQimage(self, image):

        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(
            rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
        )
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

        return p

    def detectPeople(self, color_image):

        # 600x1024 res ssd
        rgb_img = self.preProcess(color_image)

        detections = self.people_net.Detect(rgb_img)

        bboxes = self.getBBox(detections)

        return bboxes

    def detectFaces(self, peoples, color_image):
        face_detections = []

        for people, _, _ in peoples:
            (sx, sy, ex, ey) = people
            cropped = color_image[sy:ey, sx:ex]
            cv2.imshow("face", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # Detect faces
            face = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if face:
                print("face")

                (x, y, w, h) = face
                dsx = x
                dsy = y
                dex = x + w
                dey = y + h

                sx = dsx + sx
                sy = dsy + sy
                ex = dex + ex
                ey = dey + ey
                face = color_image[sy:ey, sx, ex]

                face = self.sladFaces(face)

                color_image[sy:ey, sx, ex] = face

        return color_image

    def sladFaces(self, image, blocks=3):
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        # loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)

        return image

    def alignImage(self, frames):

        # # align images
        align = rs.align(rs.stream.color)

        frameset = align.process(frames)

        # # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()

    def warning_complete(self):
        print("complete")

    def startStreaming(self):
        global depthFrames, original_frames, predicted_data, boundingBoxes, color_image2

        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()
        self.threadActive = True

        print("starting stream")
        while self.threadActive:

            try:

                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue
                # frameset = align.process(frames)

                # # Update color and depth frames:
                # aligned_depth_frame = frameset.get_depth_frame()
                # colorized_depth = np.asanyarray(
                #     colorizer.colorize(aligned_depth_frame).get_data()
                # )

                color_image = color_frame.get_data()
                color_image = np.asanyarray(color_image)

                if self.selection:
                    depthFrames.put(colorized_depth)
                    continue

                predictions = self.detectPeople(color_image)

                if len(predictions) > 0:

                    faces = self.detectFaces(predictions, color_image)

                numberOfPeople = 0

                numberOfPeople = len(predictions)

                bboxes = []
                vectors = []
                test = []

                if numberOfPeople >= 0:

                    for bbox, heightofBox, centroid in predictions:

                        bboxes.append(bbox)
                        x, y = centroid
                        vectors.append(get3d(int(x), int(y), depth_frame))
                        test.append((x, y, heightofBox))

                    pred_bbox = (bboxes, vectors)

                self.signals.people.emit(numberOfPeople)

                if pred_bbox:

                    color_image, violation = drawBox(
                        color_image, pred_bbox, self.minDistance, test
                    )

                    self.signals.violation.emit(violation)

                processed_frames.put(color_image)

            except Exception as e:
                print("Error is :", str(e))

        self.pipeline.stop()

    def stop(self):
        self.threadActive = False


class Show(QThread):
    def __init__(self, signals):
        super(Show, self).__init__()
        self.signals = signals
        self.signals.frameSelection.connect(self.updateSignal)
        self.selection = False

    def updateSignal(self, value):

        self.selection = value

    def rgbtoQimage(self, image):

        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(
            rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
        )
        p = convertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)

        return p

    @pyqtSlot()
    def run(self):
        global processed_frames, depthFrames
        print("starting show thread")
        while True:

            try:

                if self.selection:
                    image = depthFrames.get()
                else:
                    image = processed_frames.get()

                p = self.rgbtoQimage(image)
                self.signals.changePixmap.emit(p)

            except Exception as e:
                print(str(e))


class WorkerSignals(QObject):

    changePixmap = pyqtSignal(QImage)
    frameSelection = pyqtSignal(bool)
    people = pyqtSignal(int)
    min_distance = pyqtSignal(int)
    violation = pyqtSignal(set)
    finished = pyqtSignal()


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):

        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):

        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn()
        except Exception as e:

            print(str(e))

        finally:
            self.signals.finished.emit()  # Done
