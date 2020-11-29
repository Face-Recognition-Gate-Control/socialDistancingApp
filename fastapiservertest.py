from fastapi import FastAPI, Request, WebSocket
from src.utils.realsense import RealsenseCamera
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse
import numpy as np
from src.utils.post_process import *
from src.threads.helper import *
from src.detect.detect import *


camera = RealsenseCamera()
detector = Detect()
align = rs.align(rs.stream.color)
minDistance = 1
app = FastAPI()

templates = Jinja2Templates(directory="templates")

def frame_generator():
    while True:
        frames = camera.getFrame()

        color_frame = getColorImage(frames)
        depth_frame = alignImage(frames,align)

        color_image = color_frame.get_data()
        color_image = np.asanyarray(color_image)
        predictions = detector.detectPeople(color_image)

        numberOfPeople = 0

        numberOfPeople = len(predictions)

        pred_bbox = getVectorsAndBbox(predictions, depth_frame)
        if pred_bbox:


            color_image, violation = drawBox(
                color_image, pred_bbox, minDistance
            )

        frame_bytes = RealsenseCamera.frame_to_bytes(color_image)
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")



def frame_generator2():
    while True:
        frames = camera.getFrame()

        color_frame = getColorImage(frames)

        color_image = color_frame.get_data()
        color_image = np.asanyarray(color_image)

                

        face_crops, face_boxes= detector.detectFaces(color_image)
        detector.detect_face_mask(face_crops, face_boxes, color_image)
     

        frame_bytes = RealsenseCamera.frame_to_bytes(color_image)
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/")
def dashboard(request:Request):
    return templates.TemplateResponse("index.html",{
        "request":request,
        
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

@app.get("/people")
async def getPeopleCount():
    return {"count":numberOfPeople}




@app.get("/frame_streamer")
async def frame_streamer():
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame_streamer2")
async def frame_streamer2():
    return StreamingResponse(
        frame_generator2(), media_type="multipart/x-mixed-replace; boundary=frame")

