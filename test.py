import jetson.inference
import jetson.utils
import pyrealsense2 as rs
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

jsonObj = json.load(open("configrealsense.json"))
json_string = str(jsonObj).replace("'", '"')

pipeline = rs.pipeline()
rsconfig = rs.config()
cfg = pipeline.start(rsconfig)
dev = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)

# get depthscale from camera. converting distance to meter
depth_scale = cfg.get_device().first_depth_sensor().get_depth_scale()

def rgbtocuda(image):

    bgr_img = jetson.utils.cudaFromNumpy(color_image, isBGR=True)
    # convert from BGR -> RGB
    rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width, height=bgr_img.height, format="rgb8")

    jetson.utils.cudaConvertColor(bgr_img, rgb_img)

    return rgb_img

while display.IsStreaming():

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
	img = rgbtocuda(color_frame)
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))