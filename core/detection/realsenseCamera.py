import pyrealsense2 as rs


class RealsenseCamera:

    # load config file made
    # do adjustment in realsense depth quality tool
    jsonObj = json.load(open("realsense_config/configrealsense.json"))
    json_string = str(jsonObj).replace("'", '"')

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.rsconfig = rs.config
        self.camera = False
        self.init()

    def init(self):

        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

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

    def connect():
        self.camera = False

        while not self.camera:
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
                self.camera = True

    def getFrame(self):

        frame = self.pipeline.wait_for_frames()
        return frame

    def stop(self):
        self.pipeline.stop()
