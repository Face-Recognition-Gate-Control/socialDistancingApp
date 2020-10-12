# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import socket
import struct
import pickle
import numpy

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()


# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def startStream(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    SERVER = '10.22.183.75'
    PORT = 8080
    serverAddressPort = (SERVER, PORT)
    msgFromClient = "Hello UDP Server"
    bytesToSend = str.encode(msgFromClient)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    client.connect((SERVER, PORT))
    print('Socket bind complete')

    print('Socket now listening')

    data = b""
    payload_size = struct.calcsize("Q")
    print("payload_size: {}".format(payload_size))

    i = 0

    # initialize the motion detector and the total number of frames
    # read thus far

    total = 0

    # loop over frames from the video stream
    while True:

        start = time.time()
        while len(data) < payload_size:
            packet = client.recv(4*1024)  # 4K
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client.recv(4*1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments

    # start a thread that will perform motion detection
    t = threading.Thread(target=startStream, args=(
        24,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="10.22.183.75", port="8000", debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
