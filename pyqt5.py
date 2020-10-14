import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication,QSlider,QHBoxLayout,QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from pyqt.createGUI import create



class imageThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)








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
        self.initUI()
        
    def updateLabel(self, value):

        self.label1.setText(str(value))

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):

        hbox = create(self)

        btn1 = QPushButton()
        btn1.clicked.connect(self.startImage)
        btn1.resize(btn1.sizeHint())
        btn1.move(50,50)

        hbox.addWidget(btn1)

        self.setLayout(hbox)
       
        self.show()



    def startImage(self):

        if not self.stream:
            self.stream = True

            th = imageThread(self)
            th.changePixmap.connect(self.setImage)
            th.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())