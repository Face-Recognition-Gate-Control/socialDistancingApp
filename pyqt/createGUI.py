from PyQt5.QtWidgets import  QWidget, QLabel, QApplication,QSlider,QHBoxLayout,QPushButton,QVBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot 
from PyQt5.QtGui import QImage, QPixmap
import cv2
def create(self):

    
    hbox = createSlider(self)
    self.setWindowTitle(self.title)
    self.setGeometry(self.left, self.top, self.width, self.height)
    self.resize(1800, 1200)
    # create a label
    self.label = QLabel(self)
    self.label.move(280, 120)
    self.label.resize(640, 480)

    return hbox
def createSlider(self):


    hbox = QHBoxLayout()

    sld = QSlider(Qt.Horizontal, self)
    sld.setRange(0, 100)
    sld.setFocusPolicy(Qt.NoFocus)
    sld.setPageStep(5)
    
    sld.valueChanged.connect(self.updateLabel)
    self.label1 = QLabel('0', self)
    self.label1.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
    self.label1.setMinimumWidth(80)

    

    hbox.addWidget(sld)
    hbox.addSpacing(15)



    hbox.addWidget(self.label1)

    return hbox

    

def createButton(self):
    btn1 = QPushButton()
    btn1.clicked.connect()
    btn1.resize(btn1.sizeHint())
    btn1.move(50,50)
    layout = QVBoxLayout()
    layout.addWidget(btn1)
    



