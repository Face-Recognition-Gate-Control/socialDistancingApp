from PyQt5.QtCore import QObject,QRunnable,pyqtSlot,pyqtSignal,Qt,QThread
from PyQt5.QtGui import QImage
import cv2
import threading
from threading import Lock
import pyrealsense2 as rs
import os
from detect import social_distancing_config as config
import json
import numpy as np
from multiprocessing import Queue
from detect.detection import detect_people
from utils.post_process import *
import imutils

 # initilize the queues for sharing recources between processes
original_frames = Queue(maxsize=0)
detect_frames = Queue(maxsize=0)
depthFrames = Queue(maxsize=0)
predicted_data = Queue(maxsize=0)
boundingBoxes = Queue(maxsize=0)
processed_frames = Queue(maxsize=0)
preProcessed_frames =Queue(maxsize=0)



class webcamThread(QThread):
    def __init__(self,signals):
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
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.signals.changePixmap.emit(p)

                else:
                    self.stop()
            except Exception as e:

                print("release")
                cap.release()
                self.threadActive= False
                
        

        cap.release()
       
    def stop(self):
        self.threadActive = False
        
       

class realsenseThread(QThread):





    def __init__(self,signals):
        super(realsenseThread, self).__init__()
        self.signals = signals
        self.signals.frameSelection.connect(self.updateSignal)
        self.selection= False
    
    def updateSignal(self,value):
        self.selection = value
     
        
    @pyqtSlot()
    def run(self):
        
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
        camera =False

       
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


                print(e)
                print("no device connected")
                
            finally:
                print('connected')
                camera =True
                self.startStreaming()



    def startStreaming(self):
        global depthFrames,original_frames,predicted_data,boundingBoxes

       

        self.threadActive = True
        
        print('starting stream')
        while  self.threadActive:
            

            try:

                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not depth_frame or not color_frame:
                    continue
                colorizer = rs.colorizer()
                color_image = color_frame.get_data()
                color_image = np.asanyarray(color_image)
                color_image2 = np.asanyarray(color_image)
                # align images
                align = rs.align(rs.stream.color)

                frameset = align.process(frames)

                # Update color and depth frames:
                aligned_depth_frame = frameset.get_depth_frame()

                colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

                depth = np.asanyarray(aligned_depth_frame.get_data())
                
                if self.selection:

                    depthFrames.put(colorized_depth)
                
                else:
                    original_frames.put(color_image)

                    detect_frames.put(color_image2)

                if not predicted_data.empty():

                    pred_bbox = predicted_data.get()
                    numberOfPeople = len(pred_bbox)
                    self.signals.people.emit(numberOfPeople)
                    bboxes = []
                    vectors = []
                    
                
                    if numberOfPeople >= 1:
                        for (pob, bbox, centroid) in pred_bbox:
                            

                            (sx, sy, ex, ey) = bbox
                            bboxes.append(bbox)
                            w = sx + (ex - sx) / 2
                            h = sy + (ey - sy) / 2

                            vectors.append(get3d(int(w), int(h), frames))

                            boundingBoxes.put((bboxes, vectors))

            except Exception as e:
                print("Error is :", str(e))

        self.pipeline.stop()

       
    def stop(self):
        self.threadActive = False      

class Show(QThread):

    def __init__(self,signals):
        super(Show, self).__init__()
        self.signals = signals
        self.signals.frameSelection.connect(self.updateSignal)
        self.selection = False
    
    def updateSignal(self,value):

        
        self.selection = value
       
        
    @pyqtSlot()
    def run(self):
        global processed_frames,depthFrames
        print('starting show thread')
        while True:


           
            try:
               

                if self.selection:
                    image = depthFrames.get(timeout=0.1)
                else:
                    image = processed_frames.get(timeout=0.1)

                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.signals.changePixmap.emit(p)

            except Exception as e:
                print(str(e))




class PostProcess(QThread):

    def __init__(self,signals):
        super(PostProcess, self).__init__()
        self.signals = signals
        self.signals.min_distance.connect(self.updateSignal)
        self.minDistance = 1
        


    def updateSignal(self,value):

        self.minDistance = value
        
    @pyqtSlot()
    def run(self):
        global original_frames,boundingBoxes,processed_frames
        while True:


            
            
            if not original_frames.empty():

                rgb_image = original_frames.get()

                if not boundingBoxes.empty():
                    pred_bbox = boundingBoxes.get()

                    image = drawBox(rgb_image, pred_bbox,self.minDistance)

                    processed_frames.put(image)
                else:
                    processed_frames.put(rgb_image)




class PreProcess(QThread):

    def __init__(self):
        super(PreProcess, self).__init__()
        
       
       
        
    @pyqtSlot()
    def run(self):
        global detect_frames,preProcessed_frames

        while True:

            if not detect_frames.empty():

                rgb_image = detect_frames.get()

                # (h, w) = rgb_image.shape[:2]

                # frame_resized = cv2.resize(rgb_image, (300, 300))

                # blob = cv2.dnn.blobFromImage(
                #     frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
                # )
                frame = imutils.resize(rgb_image, width=700)
                preProcessed_frames.put(frame)


class WorkerSignals(QObject):
   
    changePixmap = pyqtSignal(QImage)
    frameSelection = pyqtSignal(bool)
    people = pyqtSignal(int)
    min_distance = pyqtSignal(int)



class detectionThread(QThread):

    

    def __init__(self,signals):
        super(detectionThread, self).__init__()
        self.signals = signals
        self.signals.min_distance.connect(self.updateSignal)
        self.min_Distance =1
   
    def updateSignal(self,value):
       self.min_Distance = int(value)

    @pyqtSlot()
    def run(self):
        global preProcessed_frames,predicted_data
            # # load the class labels the  model was trained on
        # labelsPath = os.path.sep.join([config.MODEL_PATH, "caffe.names"])
        # LABELS = open(labelsPath).read().strip().split("\n")
        # # derive the paths to the YOLO weights and model configuration
        # weightsPath = os.path.sep.join(
        #     [config.MODEL_PATH, "MobileNetSSD_deploy.caffemodel"]
        # )
        # configPath = os.path.sep.join([config.MODEL_PATH, "MobileNetSSD_deploy.prototxt"])

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # # load our SSD object detector trained on caffe dataset (80 classes)
        print("[INFO] loading Caffe modell from disk...")
        # # Load the Caffe model
        # net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        
        if config.USE_GPU:
        # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


        self.threadActive = True
        while self.threadActive:

            try:

                if not preProcessed_frames.empty():
                    color_image = preProcessed_frames.get()

                    results = detect_people(
                        color_image, net, ln, personIdx=LABELS.index("person")
                    )



                    #results = detect_people(color_image, net)
            

                    predicted_data.put(results)

            except Exception as e:
                print(e)
                self.threadActive=False


