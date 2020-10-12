import pyrealsense2 as rs
import cv2
from multiprocessing import Process, Queue
import numpy as np

# from detect.detection import detect_people
# from detect import social_distancing_config as config
from detect import config_caffe as config
from detect.detectCaffe import detect_people
import os
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


def predict_bbox_mp(image_queue, predicted_data):

    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # # load the class labels the  model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "caffe.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(
        [config.MODEL_PATH, "MobileNetSSD_deploy.caffemodel"]
    )
    configPath = os.path.sep.join([config.MODEL_PATH, "MobileNetSSD_deploy.prototxt"])

    # # load the COCO class labels our YOLO model was trained on
    # labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    # LABELS = open(labelsPath).read().strip().split("\n")

    # # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    # configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # print("[INFO] loading YOLO from disk...")
    # net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # # load our SSD object detector trained on caffe dataset (80 classes)
    print("[INFO] loading Caffe modell from disk...")
    # # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    # ln = net.getLayerNames()
    # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        if not image_queue.empty():
            color_image = image_queue.get()

            # results = detect_people(
            #     color_image, net, ln, personIdx=LABELS.index("person")
            # )

            results = detect_people(color_image, net)
            # results = detectHog(color_image, HOGCV)

            predicted_data.put(results)


# calcualate the distance from all the coordinates(x,y,z) from detected personns
def detectHog(frame, HOGCV):

    frame = cv2.resize(frame, (300, 300))
    print(frame.shape)
    start = datetime.datetime.now()

    # Factor for scale to original size of frame
    heightFactor = frame.shape[0] / 300.0
    widthFactor = frame.shape[1] / 300.0

    (rects, weights) = HOGCV.detectMultiScale(
        frame, winStride=(4, 4), padding=(128, 128), scale=1.05
    )

    print(rects)
    rects = np.array(
        [
            [
                int(x * widthFactor),
                int(y * heightFactor),
                int(w * widthFactor),
                int(h * heightFactor),
            ]
            for (x, y, w, h) in rects
        ]
    )
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    print("people", len(rects))

    # print(
    #     "[INFO] detection took: {}s".format(
    #         (datetime.datetime.now() - start).total_seconds()
    #     )
    # )

    # for box, weight in zip(bounding_box_cordinates, weights):
    #     print(box, weight)

    return result


def euclideanDistance(points):

    violate = set()

    for i in range(0, len(points)):

        for j in range(i + 1, len(points)):

            dist = math.sqrt(
                (points[i]["x"] - points[j]["x"]) ** 2
                + (points[i]["y"] - points[j]["y"]) ** 2
                + (points[i]["z"] - points[j]["z"]) ** 2
            )

            if dist < config.MIN_DISTANCE:

                violate.add(i)
                violate.add(j)

    return violate


def drawBox(image, predicitons):
    violation = set()

    if len(predicitons[1]) >= 2:

        violation = euclideanDistance(predicitons[1])

    for (i, (box)) in enumerate(predicitons[0]):

        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = box

        color = (255, 0, 0)
        if i in violation:
            color = (0, 0, 255)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        w = startX + (endX - startX) / 2
        h = startY + (endY - startY) / 2

        cv2.circle(image, (int(w), int(h)), 5, color, 1)

    return image


# post process the frames. draw bounding boxes of people


def postprocess_mp(bboxes, original_frames, processed_frames):

    while True:

        rgb_image = original_frames.get()

        if not bboxes.empty():
            pred_bbox = bboxes.get()

            image = drawBox(rgb_image, pred_bbox)

            processed_frames.put(image)


def Show_Image_mp(processed_image, original_image):

    print("show image thread")

    while True:

        if not processed_image.empty():
            image = processed_image.get()

            cv2.imshow("output", image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket, stream):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.clientAddress = clientAddress
        self.stream = stream
        self.lock = threading.Lock()
        self.frame = stream.get()

        print("New connection added: ", clientAddress)
        self.stopped = False

    def run(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        print("Connection from : ", self.clientAddress)

        while not self.stopped:

            self.lock.acquire()

            try:
                if not self.stream.empty():
                    print("send")
                    self.frame = self.stream.get()

                    a = pickle.dumps(self.frame, 0)
                    message = struct.pack("Q", len(a)) + a
                    self.csocket.sendall(message)
            except Exception as e:
                print(e)
            finally:
                logging.debug("Released a lock")

                self.lock.release()

        cam.release()
        print("Client at ", self.clientAddress, " disconnected...")


# starts a tcp-socket stream process
# accepts clients and spawns a new client thread for the video stream
def socketVideoStream(host, port, processed_frames):

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
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


def get3d(x, y, frames):

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_depth_intrin = (
        aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    )

    depth_pixel = [x, y]
    # In meters
    dist_to_center = aligned_depth_frame.get_distance(x, y)
    pose = rs.rs2_deproject_pixel_to_point(
        aligned_depth_intrin, depth_pixel, dist_to_center
    )

    # The (x,y,z) coordinate system of the camera is accordingly
    # Origin is at the centre of the camera
    # Positive x axis is towards right
    # Positive y axis is towards down
    # Positive z axis is into the 2d xy plane

    response_dict = {"x": pose[0], "y": pose[1], "z": pose[2]}

    return response_dict


def detect_video_realtime_mp():

    # Start processes
    p1 = Process(
        target=predict_bbox_mp,
        args=(original_frames, predicted_data),
        daemon=True,
    )

    p2 = Process(
        target=postprocess_mp,
        args=(boundingBoxes, original_frames, processed_frames),
        daemon=True,
    )

    p3 = Process(
        target=Show_Image_mp, args=(processed_frames, original_frames), daemon=True
    )

    p4 = Process(
        target=socketVideoStream,
        args=("10.22.183.75", 8080, processed_frames),
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
                print(numberOfPeople)
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
            print("Error is ", str(e))


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

    # tries to connect to realsense camera

    while True:
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

        except Exception as e:
            print(e)
            print("no device connected")

        finally:
            print("connected")

            detect_video_realtime_mp()
