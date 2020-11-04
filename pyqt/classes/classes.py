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
from core.detection import config_caffe as config
import json
import numpy as np
from multiprocessing import Queue
from core.detection.detectCaffe import detect_people
from utils.post_process import *
import imutils
import simpleaudio as sa
import time
from pyqt.classes.detect import *
from core.detection.realsenseCamera import RealsenseCamera

# vfrom server.pyClient import ClientPy
from imutils.object_detection import non_max_suppression


# initilize the queues for sharing recources between processes
original_frames = Queue(maxsize=0)
depthFrames = Queue(maxsize=0)
predicted_data = Queue(maxsize=0)
boundingBoxes = Queue(maxsize=0)
processed_frames = Queue(maxsize=0)
preProcessed_frames = Queue(maxsize=0)
client_data = Queue(maxsize=0)
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
        self.detector = Detect()
        self.camera = RealsenseCamera()
        # self.client = ClientPy()

    def updateSignal(self, value):
        self.selection = value

    def updateSignal2(self, value):

        self.minDistance = value

    @pyqtSlot()
    def run(self):

        while not self.camera.camera:
            self.camera.connect()

        self.startStreaming()

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

    def alignImage(self, frames):

        frameset = self.align.process(frames)

        # # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()

        return aligned_depth_frame

    def warning_complete(self):
        print("complete")

    def getVectorsAndBbox(self, predictions, depth_frame):
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

    def startStreaming(self):
        global depthFrames, original_frames

        self.threadActive = True

        print("starting stream")
        while self.threadActive:

            try:

                frames = self.camera.getFrame()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # depth_frame = self.alignImage(frames)
                # colorized_depth = np.asanyarray(
                #     colorizer.colorize(aligned_depth_frame).get_data()
                # )

                color_image = color_frame.get_data()
                color_image = np.asanyarray(color_image)

                if self.selection:
                    depthFrames.put(colorized_depth)
                    continue

                predictions = self.detector.detectPeople(color_image)

                faces = self.detector.detectFaces(predictions, color_image)

                numberOfPeople = 0

                numberOfPeople = len(predictions)

                pred_bbox = self.getVectorsAndBbox(predictions, depth_frame)

                self.signals.people.emit(numberOfPeople)

                if pred_bbox:

                    color_image, violation = drawBox(
                        color_image, pred_bbox, self.minDistance
                    )

                    self.signals.violation.emit(violation)

                processed_frames.put(color_image)

            except Exception as e:
                print("Error is :", str(e))

        self.camera.stop()

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
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

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
