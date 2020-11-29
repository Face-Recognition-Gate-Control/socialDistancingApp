
import pyrealsense2 as rs
from multiprocessing import Queue
import numpy as np

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
from ..detect.detect import *
from .helper import *

class MaskDetection(QThread):
    def __init__(self, signals, cap,imageQueue,lock,Detect):
        super(MaskDetection, self).__init__()
        self.camera =cap
        self.align = rs.align(rs.stream.color)
        self.queue = imageQueue
        self.signals = signals
        self.lock = lock
        self.detector = Detect
        self.align = rs.align(rs.stream.color)
        self.signals.tab_selection.connect(self.updateTab)
        self.tab = 0
        self.frame = signals.maskDetection_frame
    
    
    
    
    
    
    
    def updateTab(self,value):
        print("from mask",value)
        self.tab = value
    
    
    @pyqtSlot()
    def run(self):
        
        
    
        self.threadActive = True



        

        print("starting stream")
        while self.threadActive:
            if self.tab ==1:
                self.setPriority(QThread.TimeCriticalPriority)
                
                self.lock.lockForRead()

                #tic = time.perf_counter()
                try:
                    
                    frames = self.camera.getFrame()
                    self.lock.unlock()
                    # time.sleep(0.5)
                    
                    color_frame = frames.get_color_frame()
                    

                    if not color_frame:
                        continue

                    

                    color_image = color_frame.get_data()
                    color_image = np.asanyarray(color_image)

                    
                   

                    face_crops, face_boxes =self.detector.detectFaces(color_image)

                    self.detector.detect_face_mask(face_crops, face_boxes, color_image)

                    #self.queue.put(color_image)
                    p = rgbtoQimage(color_image)
                   
                    self.frame.emit(p)

                    toc = time.perf_counter()

                
            
                except Exception as e:
                    print("Error is :", str(e))
                    #self.quit()
            
            
             

       
        
        #self.camera.stop()
        #self.quit()

    def stop(self):
        self.threadActive = False
