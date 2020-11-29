
import pyrealsense2 as rs
import numpy as np
from ..utils.post_process import *
from .helper import *
from ..detect.detect import *
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
    QRunnable,
    QThreadPool,
    QObject,
    QTimer,
)
import time


class SocialDistancing(QThread):
    def __init__(self, signals,cap,imageQueue,lock,Detect):
        super(SocialDistancing, self).__init__()
        self.align = rs.align(rs.stream.color)
        self.queue = imageQueue
        self.signals = signals
        self.camera = cap
        self.lock = lock
        self.detector = Detect
        self.align = rs.align(rs.stream.color)
        self.minDistance = 1
        self.signals.min_distance.connect(self.updateSignal)
        self.signals.tab_selection.connect(self.updateTab)
        self.tab = 0
        self.frame = signals.distance_frame
        
    def updateSignal(self, value):
        self.minDistance = value
    
    def updateTab(self,value):
        
        self.tab = value
    
    
    @pyqtSlot()
    def run(self):
        

     
        
    

        self.threadActive = True



        

        print("starting stream")
        while self.threadActive:
            if self.tab ==0:
                self.setPriority(QThread.TimeCriticalPriority)
                self.lock.lockForRead()
                try:
                    
                    frames = self.camera.getFrame()
                    self.lock.unlock()
                    # time.sleep(0.5)
                    
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    depth_frame = alignImage(frames,self.align)

                    color_image = color_frame.get_data()
                    color_image = np.asanyarray(color_image)
                    
                        

                    predictions = self.detector.detectPeople(color_image)

                    numberOfPeople = 0

                    numberOfPeople = len(predictions)

                    pred_bbox = getVectorsAndBbox(predictions, depth_frame)

                    self.signals.people.emit(numberOfPeople)

                    if pred_bbox:


                        color_image, violation = drawBox(
                            color_image, pred_bbox, self.minDistance
                        )

                        self.signals.violation.emit(violation)









                    #self.queue.put(color_image)
                    p = rgbtoQimage(color_image)
                   
                    self.frame.emit(p)
                
                


            
                except Exception as e:

                    print("Error is :", str(e))
                    #self.quit()
            
            
                
                

        
        
        #self.camera.stop()
        #self.quit()

    def stop(self):
        self.threadActive = False
