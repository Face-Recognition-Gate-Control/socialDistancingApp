import sys
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
from PyQt5.QtGui import QImage, QPixmap, QColor
from pyqt.createGUI import create
import time
import threading, queue
from multiprocessing import Queue
from pyqt.classes.classes import *
from pyqt.gui.ui_splash_screen import Ui_SplashScreen
from pyqt.gui.ui_main import Ui_MainWindow

## ==> GLOBALS
counter = 0


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.sound = False
        self.camerastream = False
        self.signals = WorkerSignals()
        self.ui.camera.clicked.connect(self.test)
        self.ui.distance.valueChanged.connect(self.updateDistance)
        self.ui.radioButton.toggled.connect(lambda: self.btnstate(self.ui.radioButton))
        self.threadpool = QThreadPool()
        self.violations = 0

        self.image = realsenseThread(self.signals)
        self.image.signals.people.connect(self.setValue)
        self.image.signals.violation.connect(self.violation)

        self.showImage = Show(self.signals)
        self.showImage.signals.changePixmap.connect(self.setImage)

        self.showImage.start()

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

    def violation(self, value):

        if len(value) > 0 and not self.sound:

            self.violations += 1
            self.ui.status.setText(str(self.violations))
            self.sound = True
            self.playSound()

    def test(self):

        try:
            if not self.camerastream:

                self.image.start()
                self.camerastream = True

            else:
                self.image.threadActive = False

                self.camerastream = False
        except Exception as e:

            pass

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
            self.main = MainWindow()
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SplashScreen()
    sys.exit(app.exec_())