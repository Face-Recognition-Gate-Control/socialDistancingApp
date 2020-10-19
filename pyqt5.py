
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication,QSlider,QHBoxLayout,QPushButton,QMainWindow,QGraphicsDropShadowEffect
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot,QRunnable,QThreadPool,QObject,QTimer
from PyQt5.QtGui import QImage, QPixmap,QColor
from pyqt.createGUI import create
import time
import threading
from pyqt.classes.classes import *

from pyqt.gui.ui_splash_screen import Ui_SplashScreen
from pyqt.gui.ui_main import Ui_MainWindow

## ==> GLOBALS
counter = 0

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.camerastream = False
        self.signals = WorkerSignals()
        self.ui.camera.clicked.connect(self.test)
        self.ui.distance.valueChanged.connect(self.updateDistance)
        self.threadpool = QThreadPool()
        self.detect = detectionThread(self.signals)
        self.threadpool.start(self.detect)
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        



    
    def updateDistance(self, value):

        self.ui.distance_value.setText(str(value))
        self.signals.min_distance.emit(int(self.ui.distance_value.text()))

    def test(self):

        try:
            if(not self.camerastream):
                
                self.image = imageThread(self.signals)
                self.image.signals.changePixmap.connect(self.setImage)
                self.threadpool.start(self.image)
                self.camerastream = True
            
            else:
                self.image.stop()
                
                self.camerastream = False
        except Exception as e:

            print(e)  


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.ui.cameraStream.setPixmap(QPixmap.fromImage(image))

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
        QTimer.singleShot(1500, lambda: self.ui.label_description.setText("<strong>LOADING</strong> Threads"))
        QTimer.singleShot(3000, lambda: self.ui.label_description.setText("<strong>LOADING</strong> Magic"))


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
            self.main = MainWindow()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1










class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.stream = False
        self.number_of_people = 0 
        self.minDistance =0
        self.signals = WorkerSignals()
        self.initUI()
    
    





    def updateLabel(self, value):

        self.label1.setText(str(value))
        self.signals.min_distance.emit(int(self.label1.text()))
    
    



    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))



    def test2(self):
        detect = detectionThread(self.signals)
        
        self.threadpool.start(detect)

    def test(self):
        image = imageThread(self.signals)
        image.signals.changePixmap.connect(self.setImage)
        self.threadpool.start(image) 



    def initUI(self):

        hbox = create(self)
        btn1 = QPushButton()
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.test2()
        
        btn1.clicked.connect(self.test)
        btn1.resize(btn1.sizeHint())
        btn1.move(50,50)

        hbox.addWidget(btn1)

        self.setLayout(hbox)

        #self.startDetect()       
        self.show()



  
        

    






if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SplashScreen()
    sys.exit(app.exec_())