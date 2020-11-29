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

from PyQt5.QtGui import QImage

import cv2


class WorkerSignals(QObject):




    distance_frame = pyqtSignal(QImage)
    maskDetection_frame = pyqtSignal(QImage)
    frameSelection = pyqtSignal(bool)
    people = pyqtSignal(int)
    min_distance = pyqtSignal(int)
    violation = pyqtSignal(set)
    finished = pyqtSignal()
    tab_selection = pyqtSignal(int)
  


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):

        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):

        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn()
        except Exception as e:

            print(str(e))

        finally:
            self.signals.finished.emit()  # Done



class Show(QThread):
    def __init__(self, signal,queue):
        super(Show, self).__init__()
        self.frame = signal
        self.selection = False
        self.queue =queue
        
       

    def updateSignal(self, value):

        self.selection = value

    def rgbtoQimage(self, image):

        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(
            rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
        )
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

        return p

    @pyqtSlot()
    def run(self):

       
        print("starting show thread")
        while True:

            try:

                if not self.queue.empty():
                    

                    image = self.queue.get()
                    
                    p = self.rgbtoQimage(image)
                   
                    self.frame.emit(p)
                    

                    

            except Exception as e:
                print(str(e))
        self.quit()