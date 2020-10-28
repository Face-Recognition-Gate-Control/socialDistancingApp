from PyQt5.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal, Qt, QThread
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


class webcamThread(QThread):
    def __init__(self, signals):
        super(webcamThread, self).__init__()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        global original_frames
        cap = cv2.VideoCapture(0)
        self.threadActive = True
        while self.threadActive:
            try:
                ret, frame = cap.read()
                if ret:
                    # https://stackoverflow.com/a/55468544/6622587
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(
                        rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                    )
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.signals.changePixmap.emit(p)

                else:
                    self.stop()
            except Exception as e:

                print("release")
                cap.release()
                self.threadActive = False

        cap.release()

    def stop(self):
        self.threadActive = False


class realsenseThread(QThread):
    def __init__(self, signals):
        super(realsenseThread, self).__init__()
        self.signals = signals
        self.signals.frameSelection.connect(self.updateSignal)
        self.signals.min_distance.connect(self.updateSignal2)
        self.selection = False
        self.minDistance = 1

    def updateSignal(self, value):
        self.selection = value

    def updateSignal2(self, value):

        self.minDistance = value

    @pyqtSlot()
    def run(self):

        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.55)

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

            results.append(bbox)

        return results

    def preProcess(self, color_image):

        bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
        # convert from BGR -> RGB
        rgb_img = jetson.utils.cudaAllocMapped(
            width=bgr_img.width, height=bgr_img.height, format="rgb8"
        )

        jetson.utils.cudaConvertColor(bgr_img, rgb_img)

        return rgb_img

    def detect(self, color_image):

        rgb_img = self.preProcess(color_image)

        detections = self.net.Detect(rgb_img)

        bboxes = self.getBBox(detections)

        return bboxes

    def startStreaming(self):
        global depthFrames, original_frames, predicted_data, boundingBoxes, color_image2

        self.threadActive = True

        print("starting stream")
        while self.threadActive:

            try:

                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not depth_frame or not color_frame:
                    continue
                colorizer = rs.colorizer()
                color_image = color_frame.get_data()
                color_image = np.asanyarray(color_image)

                # align images
                align = rs.align(rs.stream.color)

                frameset = align.process(frames)

                # Update color and depth frames:
                aligned_depth_frame = frameset.get_depth_frame()

                colorized_depth = np.asanyarray(
                    colorizer.colorize(aligned_depth_frame).get_data()
                )

                if self.selection:
                    depthFrames.put(colorized_depth)
                    continue

                predictions = self.detect(color_image)

                numberOfPeople = 0

                numberOfPeople = len(predictions)

                bboxes = []
                vectors = []

                if numberOfPeople >= 0:

                    for bbox in predictions:

                        (sx, sy, ex, ey) = bbox
                        bboxes.append(bbox)
                        w = sx + (ex - sx) / 2
                        h = sy + (ey - sy) / 2

                        vectors.append(get3d(int(w), int(h), aligned_depth_frame))

                    pred_bbox = (bboxes, vectors)

                self.signals.people.emit(numberOfPeople)

                if pred_bbox:

                    color_image, violation = drawBox(
                        color_image, pred_bbox, self.minDistance
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

    @pyqtSlot()
    def run(self):
        global processed_frames, depthFrames
        print("starting show thread")
        while True:

            try:

                if self.selection:
                    image = depthFrames.get(timeout=0.01)
                else:
                    image = processed_frames.get(timeout=0.01)

                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                )
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.signals.changePixmap.emit(p)

            except Exception as e:
                print(str(e))


class PostProcess(QThread):
    def __init__(self, signals):
        super(PostProcess, self).__init__()
        self.signals = signals
        self.signals.min_distance.connect(self.updateSignal)
        self.minDistance = 1

    def updateSignal(self, value):

        self.minDistance = value

    @pyqtSlot()
    def run(self):
        global original_frames, boundingBoxes, processed_frames
        while True:

            rgb_image = original_frames.get()

            if not boundingBoxes.empty():

                pred_bbox = boundingBoxes.get()

                image, violation = drawBox(rgb_image, pred_bbox, self.minDistance)

                self.signals.violation.emit(violation)
                processed_frames.put(image)
            else:
                processed_frames.put(rgb_image)


class PreProcess(QThread):
    def __init__(self):
        super(PreProcess, self).__init__()

    @pyqtSlot()
    def run(self):
        global detect_frames, preProcessed_frames

        while True:

            if not detect_frames.empty():

                rgb_image = detect_frames.get()

                frame_resized = cv2.resize(rgb_image, (300, 300))

                blob = cv2.dnn.blobFromImage(
                    frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
                )
                # frame = imutils.resize(rgb_image, width=700)
                preProcessed_frames.put(blob)


class WorkerSignals(QObject):

    changePixmap = pyqtSignal(QImage)
    frameSelection = pyqtSignal(bool)
    people = pyqtSignal(int)
    min_distance = pyqtSignal(int)
    violation = pyqtSignal(set)
    finished = pyqtSignal()


class detectionThread(QThread):
    def __init__(self, signals):
        super(detectionThread, self).__init__()
        self.signals = signals
        self.signals.min_distance.connect(self.updateSignal)
        self.min_Distance = 1

    def updateSignal(self, value):
        self.min_Distance = int(value)

    def preProcess(self, color_image):

        bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
        # convert from BGR -> RGB
        rgb_img = jetson.utils.cudaAllocMapped(
            width=bgr_img.width, height=bgr_img.height, format="rgb8"
        )

        jetson.utils.cudaConvertColor(bgr_img, rgb_img)

        return rgb_img

    def listToString(self, s):

        # initialize an empty string
        str1 = " "

        # return string
        return str1.join(s)

    def getBBox(self, detections):
        results = []

        for detection in detections:
            bbox = (
                int(detection.Left),
                int(detection.Top),
                int(detection.Right),
                int(detection.Bottom),
            )

            results.append(bbox)

        return results

    @pyqtSlot()
    def run(self):
        global color_image2, predicted_data

        net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)

        self.threadActive = True
        while self.threadActive:

            detect_lock.acquire()
            try:

                if len(color_image2) > 0:

                    color_image = color_image2
                    rgb_img = self.preProcess(color_image)
                    detections = net.Detect(rgb_img)

                    bboxes = self.getBBox(detections)
                    if len(bboxes) > 0:
                        predicted_data.put(bboxes)

            except Exception as e:
                print(str(e))
                self.threadActive = False
            finally:

                detect_lock.release()


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
