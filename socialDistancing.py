import pyrealsense2 as rs
import cv2
from multiprocessing import Process, Queue
import numpy as np

from flask import Response
from flask import Flask
from flask import render_template
# from detect.detection import detect_people
# from detect import social_distancing_config as config
from detect import config_caffe as config
from detect.detectCaffe import detect_people
import os
from threading import Thread
import imutils
from imutils.object_detection import non_max_suppression
import json
import math
import threading
import pickle
import socket
import logging
import datetime
import struct

from utils.calculation import euclideanDistance 
from utils.post_process import  drawBox,get3d






def predict_bbox(preProcessed_frames, predicted_data):

  

    # # load the class labels the  model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "caffe.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(
        [config.MODEL_PATH, "MobileNetSSD_deploy.caffemodel"]
    )
    configPath = os.path.sep.join([config.MODEL_PATH, "MobileNetSSD_deploy.prototxt"])


    # # load our SSD object detector trained on caffe dataset (80 classes)
    print("[INFO] loading Caffe modell from disk...")
    # # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

    

  

    while True:
        if not preProcessed_frames.empty():
            color_image = preProcessed_frames.get()

            # results = detect_people(
            #     color_image, net, ln, personIdx=LABELS.index("person")
            # )

            results = detect_people(color_image, net)
            

            predicted_data.put(results)









# post process the frames. draw bounding boxes of people


def postprocess(bboxes, preProcessed_frames, processed_frames):


    while True:

        if not preProcessed_frames.empty():

            rgb_image = preProcessed_frames.get()

            if not bboxes.empty():
                pred_bbox = bboxes.get()

                image = drawBox(rgb_image, pred_bbox)

                processed_frames.put(image)



def preProcess(original_frames,preProcessed_frames):

    while True:

        if not original_frames.empty():
            rgb_image = original_frames.get()

            (h, w) = rgb_image.shape[:2]

            frame_resized = cv2.resize(rgb_image, (300, 300))

            blob = cv2.dnn.blobFromImage(
                frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
            )
            preProcessed_frames.put(blob)





def Show_Image(processed_image, original_image):

    print("show image thread")

    while True:

        if not processed_image.empty():
            image = processed_image.get()

            cv2.imshow("output", image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break




# starts a tcp-socket stream process
# accepts clients and spawns a new client thread for the video stream
def socketVideoStream(host, port, processed_frames):

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    lock = threading.Lock()

    print("Server started")
    print("Waiting for client request..")
    while True:
        server.listen(5)
        clientsocket, clientAddress = server.accept()
        newthread = ClientThread(clientAddress, clientsocket, processed_frames)

        newthread.start()
    server.close()






def detect_video_realtime():

    # Start processes
    p1 = Process(
        target=predict_bbox,
        args=(preProcessed_frames, predicted_data),
        daemon=True,
    )

    p2 = Process(
        target=postprocess,
        args=(boundingBoxes, original_frames, processed_frames),
        daemon=True,
    )

    p3 = Process(
        target=Show_Image, args=(processed_frames, original_frames), daemon=True
    )

    p4 = Process(
        target=preProcess,
        args=(original_frames,preProcessed_frames),
        daemon=True,
    )

    

  

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    

    while True:

        try:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = color_frame.get_data()
            color_image = np.asanyarray(color_image)
            # align images
            align = rs.align(rs.stream.color)

            frameset = align.process(frames)

            # Update color and depth frames:
            aligned_depth_frame = frameset.get_depth_frame()

            depth = np.asanyarray(aligned_depth_frame.get_data())

            depthFrames.put(depth)

            original_frames.put(color_image)

            if not predicted_data.empty():

                pred_bbox = predicted_data.get()
                numberOfPeople = len(pred_bbox)
                bboxes = []
                vectors = []
               
                if numberOfPeople >= 2:
                    for bbox in pred_bbox:

                        (sx, sy, ex, ey) = bbox
                        bboxes.append(bbox)
                        w = sx + (ex - sx) / 2
                        h = sy + (ey - sy) / 2

                        vectors.append(get3d(int(w), int(h), frames))

                boundingBoxes.put((bboxes, vectors))

        except Exception as e:
            print("Error is :", str(e))






if __name__ == "__main__":

   

   


    # load config file made
    # do adjustment in realsense depth quality tool
    jsonObj = json.load(open("configrealsense.json"))
    json_string = str(jsonObj).replace("'", '"')

    pipeline = rs.pipeline()
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
    # tries to connect to realsense camera

    while not camera:
        try:

            print("connecting to realsense")
            cfg = pipeline.start(rsconfig)
            dev = cfg.get_device()
            advnc_mode = rs.rs400_advanced_mode(dev)
            advnc_mode.load_json(json_string)

            # get depthscale from camera. converting distance to meter
            depth_scale = cfg.get_device().first_depth_sensor().get_depth_scale()

            # initilize the queues for sharing recources between processes
            original_frames = Queue()
            depthFrames = Queue()
            predicted_data = Queue()
            boundingBoxes = Queue()
            processed_frames = Queue()
            preProcessed_frames =Queue()

           

        except Exception as e:
            print(e)
            print("no device connected")

        finally:
            print("connected")
            camera = True

         

            detect_video_realtime()

            
            
            
            
            
