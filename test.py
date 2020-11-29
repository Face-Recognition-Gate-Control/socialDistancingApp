import jetson.inference
import jetson.utils
import pyrealsense2 as rs
import json
import numpy as np


def rgbtocuda(color_image):

    bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
    # convert from BGR -> RGB
    rgb_img = jetson.utils.cudaAllocMapped(
        width=bgr_img.width, height=bgr_img.height, format="rgb8"
    )

    jetson.utils.cudaConvertColor(bgr_img, rgb_img)

    return rgb_img


def test():

    while True:

        try:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue
            color_image = color_frame.get_data()
            color_image = np.asanyarray(color_image)
            img = rgbtocuda(color_image)
            detections = net.Detect(img)
            display.Render(img)
            display.SetStatus(
                "Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())
            )

        except Exception as e:

            print(e)


def startApp():

    camera = False
    # tries to connect to realsense camera

    while not camera:
        try:

            print("connecting to realsense")
            cfg = pipeline.start(rsconfig)
            dev = cfg.get_device()
            advnc_mode = rs.rs400_advanced_mode(dev)
            advnc_mode.load_json(json_string)

        except Exception as e:
            print(e)
            print("no device connected")

        finally:
            print("connected")
            camera = True
            test()


# check to see if this is the main thread of execution
if __name__ == "__main__":

    net = jetson.inference.detectNet("pednet", threshold=0.75)
    camera = jetson.utils.videoSource("csi://0")  # '/dev/video0' for V4L2
    display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file

    # load config file made
    # do adjustment in realsense depth quality tool
    jsonObj = json.load(open("realsense_config/configrealsense.json"))
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

    startApp()