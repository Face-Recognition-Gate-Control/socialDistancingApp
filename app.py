import sys  # needed for `argv` tranfer to `QApplication`
from PyQt5 import QtWidgets
from Gui.design.ui_main import Ui_MainWindow
from src.utils.realsense import RealsenseCamera
from Gui.design.ui_splash_screen import Ui_SplashScreen
from PyQt5.QtGui import QImage, QPixmap, QColor
from Gui.classes.classes import WorkerSignals,Worker,Show
from src.threads.socialDistancing import SocialDistancing
from src.threads.maskDetection import MaskDetection
from multiprocessing import Queue
from src.detect.detect import Detect
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QApplication,
    QSlider,
    QHBoxLayout,
    QPushButton,
    QMainWindow,
    QGraphicsDropShadowEffect,
)
from threading import Lock
import simpleaudio as sa

from PyQt5.QtCore import (
QThread,
Qt,
pyqtSignal,
pyqtSlot,
QRunnable,
QThreadPool,
QObject,
QTimer,
QMutex,
QReadWriteLock
)

## ==> GLOBALS
counter = 0
class Gui(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.lock = QReadWriteLock
        self.cap = RealsenseCamera()
        self.detector = Detect() 
        
        
        self.sound = False
        self.camerastream = False
        self.signals = WorkerSignals()
        self.objThread = QThread()
        self.ui.camera.clicked.connect(self.test)
        self.ui.distance.valueChanged.connect(self.updateDistance)
        self.threadpool = QThreadPool()
        self.violations = 0
        self.commandQueue = Queue()
        self.mask_frame_queue = Queue(maxsize=0)
        self.distance_frame_queue = Queue(maxsize=0)
        self.threads =[]
        self.ui.camera.clicked.connect(self.test)
        self.ui.tabWidget.currentChanged.connect(self.tab)

        #Init people detection stream
        self.social_dist = SocialDistancing(self.signals,self.cap,self.distance_frame_queue,self.lock,self.detector)
        self.social_dist.signals.people.connect(self.setValue)
        self.social_dist.signals.violation.connect(self.warning)
        
        
        #init mask_detection stream
        self.mask_det = MaskDetection(self.signals, self.cap,self.mask_frame_queue,self.lock,self.detector)
        

        #init 
        self.show_mask = Show(self.signals.maskDetection_frame,self.mask_frame_queue)
        self.show_mask.frame.connect(self.setMaskDetectionStream)
        #self.show_mask.start()
        
        
        self.show_distance = Show(self.signals.distance_frame, self.distance_frame_queue)
        self.show_distance.frame.connect(self.setDistanceStream)
        #self.show_distance.start()
        
        
    

    
    
    
    
    
    def tab(self,value):
        self.signals.tab_selection.emit(value)

    
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.commandQueue.put("up")
        if event.key() == Qt.Key_S:
            self.commandQueue.put("down")
        if event.key() == Qt.Key_A:
            self.commandQueue.put("left")
        if event.key() == Qt.Key_D:
            self.commandQueue.put("right")

    def playSound(self):

        worker = Worker(self.testLyd)
        worker.signals.finished.connect(self.warning_complete)
        self.threadpool.start(worker)

    def warning_complete(self):
        self.sound = False

    def testLyd(self):
        filename = "record.wav"
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until sound has finished playing

    def btnstate(self, b):

        if b.isChecked():
            self.signals.frameSelection.emit(True)
        else:
            self.signals.frameSelection.emit(False)

    def updateDistance(self, value):

        self.ui.distance_value.setText(str(value))
        self.signals.min_distance.emit(int(self.ui.distance_value.text()))

    def setValue(self, value):
        self.ui.people.setText(str(value))

    def warning(self, value):

        if len(value) > 0 and not self.sound:

            self.violations += 1
            self.ui.status.setText(str(self.violations))
            self.sound = True
            self.playSound()
            

    def test(self):

        try:
            if not self.camerastream:

                self.social_dist.start()
                self.mask_det.start()
                self.camerastream = True

            else:
                # self.social_dist.stop()
                # self.mask_det.stop()

                self.camerastream = False
        except Exception as e:
            print(str(e))
            pass

    @pyqtSlot(QImage)
    def setDistanceStream(self, image):
        self.ui.stream_1.setPixmap(QPixmap.fromImage(image))

    
    @pyqtSlot(QImage)
    def setMaskDetectionStream(self, image):
        self.ui.stream_2.setPixmap(QPixmap.fromImage(image))









class SplashScreen(QMainWindow):
    def __init__(self):

        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        ## UI ==> INTERFACE CODES
        ########################################################################

        ## REMOVE TITLE BAR
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        ## DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        ## QTIMER ==> START
        self.timer = QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(35)

        self.ui.label_description.setText("<strong>WELCOME</strong> TO MY APPLICATION")

        # Change Texts
        QTimer.singleShot(
            1500,
            lambda: self.ui.label_description.setText(
                "<strong>LOADING</strong> Threads"
            ),
        )
        QTimer.singleShot(
            3000,
            lambda: self.ui.label_description.setText("<strong>LOADING</strong> Magic"),
        )

        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    ## ==> APP FUNCTIONS
    ########################################################################
    def progress(self):

        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = Gui()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SplashScreen()
    sys.exit(app.exec_())