from PyQt5.QtCore import QObject,QRunnable,pyqtSlot,pyqtSignal,Qt,QThread
from PyQt5.QtGui import QImage
import cv2
import threading

class imageThread(QRunnable):
    def __init__(self,signals):
        super(imageThread, self).__init__()
        self.signals = WorkerSignals()
        self.threadActive = True
       
        
    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(0)
        while self.threadActive:
            try:
                ret, frame = cap.read()
                if ret:
                    # https://stackoverflow.com/a/55468544/6622587
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.signals.changePixmap.emit(p)

                else:
                    self.stop()
            except Exception as e:

                print("release")
                cap.release()
                
        

        cap.release()
       
    def stop(self):
        self.threadActive = False
       

          
        


class WorkerSignals(QObject):
   
    changePixmap = pyqtSignal(QImage)
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    min_distance = pyqtSignal(int)



class detectionThread(QRunnable):

    

    def __init__(self,signals):
        super(detectionThread, self).__init__()
        self.signals = signals
        self.signals.min_distance.connect(self.updateSignal)
        self.min_Distance =0
   
    def updateSignal(self,value):
       self.min_Distance = int(value)

    @pyqtSlot()
    def run(self):

        while 1:

            print(self.min_Distance)

