import pyrealsense2 as rs
import json
import cv2




class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.rsconfig = rs.config()
        self.camera = False
        self.init()
        

    def init(self):

        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        # load config file made
        # do adjustment in realsense depth quality tool
        jsonObj = json.load(open("realsense_config/configrealsense.json"))
        self.json_string = str(jsonObj).replace("'", '"')

        freq = int(jsonObj["stream-fps"])
        print("W: ", int(jsonObj["stream-width"]))
        print("H: ", int(jsonObj["stream-height"]))
        print("FPS: ", int(jsonObj["stream-fps"]))
        self.rsconfig.enable_stream(
            rs.stream.depth,
            int(jsonObj["stream-width"]),
            int(jsonObj["stream-height"]),
            rs.format.z16,
            int(jsonObj["stream-fps"]),
        )
        self.rsconfig.enable_stream(
            rs.stream.color,
            int(jsonObj["stream-width"]),
            int(jsonObj["stream-height"]),
            rs.format.bgr8,
            int(jsonObj["stream-fps"]),
        )

        self.connect()
    
    @staticmethod
    def frame_to_bytes(frame):
        """ converts a numpy.ndarray into bytes
        Args:
            frame ([numpy.ndarray]): frame
        Returns:
            [bytes]: [output buffer]
        """
        _, frame_buffer = cv2.imencode('.jpg', frame)
        return frame_buffer.tobytes()
   
    def connect(self):
        self.camera = False

        while not self.camera:
            try:
                print("connecting to realsense")
                cfg = self.pipeline.start(self.rsconfig)
                dev = cfg.get_device()
                advnc_mode = rs.rs400_advanced_mode(dev)
                advnc_mode.load_json(self.json_string)
                


                # get depthscale from camera. converting distance to meter
                self.depth_scale = cfg.get_device().first_depth_sensor().get_depth_scale()

                for x in range(10):

                    self.pipeline.wait_for_frames()

            except Exception as e:

                print(str(e))
                print("no device connected")

            finally:
                print("connected")
                self.camera = True

    def getFrame(self):

        frame = self.pipeline.wait_for_frames()
        return frame

    
    def getColor(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()



    def stop(self):
        self.pipeline.stop()
