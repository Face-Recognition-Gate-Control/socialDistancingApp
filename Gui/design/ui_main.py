# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setStyleSheet("QFrame {    \n"
"    background-color: rgb(38,42,54);    \n"
"    color: rgb(220, 220, 220);\n"
"    border-radius: 10px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(-1, -1, 1301, 751))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.tabWidget = QtWidgets.QTabWidget(self.frame_3)
        self.tabWidget.setGeometry(QtCore.QRect(260, 60, 681, 531))
        self.tabWidget.setStyleSheet("QLabel {    \n"
"    background-color: rgb(38,42,54);    \n"
"    color: rgb(220, 220, 220);\n"
"    border-radius: 10px;\n"
"}")
        self.tabWidget.setObjectName("tabWidget")
        self.Tab_1 = QtWidgets.QWidget()
        self.Tab_1.setObjectName("Tab_1")
        self.frame_2 = QtWidgets.QFrame(self.Tab_1)
        self.frame_2.setGeometry(QtCore.QRect(-10, -11, 851, 701))
        self.frame_2.setStyleSheet("QFrame {    \n"
"    background-color: rgb(38,42,54);    \n"
"    color: rgb(220, 220, 220);\n"
"    border-radius: 10px;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.stream_1 = QtWidgets.QLabel(self.frame_2)
        self.stream_1.setGeometry(QtCore.QRect(20, 20, 640, 480))
        self.stream_1.setText("")
        self.stream_1.setObjectName("stream_1")
        self.tabWidget.addTab(self.Tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.frame = QtWidgets.QFrame(self.tab_2)
        self.frame.setGeometry(QtCore.QRect(-20, -50, 701, 561))
        self.frame.setStyleSheet("QFrame {    \n"
"    background-color: rgb(38,42,54);    \n"
"    color: rgb(220, 220, 220);\n"
"    border-radius: 10px;\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.stream_2 = QtWidgets.QLabel(self.frame)
        self.stream_2.setGeometry(QtCore.QRect(40, 60, 640, 480))
        self.stream_2.setText("")
        self.stream_2.setObjectName("stream_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(440, 0, 311, 71))
        self.label_3.setObjectName("label_3")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(90, 620, 190, 79))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.distance_value = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.distance_value.setObjectName("distance_value")
        self.verticalLayout_3.addWidget(self.distance_value, 0, QtCore.Qt.AlignHCenter)
        self.distance = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.distance.setMaximum(5)
        self.distance.setSingleStep(1)
        self.distance.setProperty("value", 1)
        self.distance.setOrientation(QtCore.Qt.Horizontal)
        self.distance.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.distance.setObjectName("distance")
        self.verticalLayout_3.addWidget(self.distance)
        self.label_5 = QtWidgets.QLabel(self.frame_3)
        self.label_5.setGeometry(QtCore.QRect(430, 620, 281, 71))
        self.label_5.setObjectName("label_5")
        self.status = QtWidgets.QLabel(self.frame_3)
        self.status.setGeometry(QtCore.QRect(660, 650, 47, 13))
        self.status.setText("")
        self.status.setObjectName("status")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(970, 610, 211, 81))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 9, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.people = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.people.setFont(font)
        self.people.setObjectName("people")
        self.verticalLayout_2.addWidget(self.people, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.camera = QtWidgets.QPushButton(self.frame_3)
        self.camera.setGeometry(QtCore.QRect(1084, 60, 101, 23))
        self.camera.setObjectName("camera")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab_1), _translate("MainWindow", "Distance"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Mask"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:22pt; font-weight:600;\">Social Distancing</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Minimum distance meter</span></p></body></html>"))
        self.distance_value.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt;\">1</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">Number of violations</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "Number of people"))
        self.people.setText(_translate("MainWindow", "0"))
        self.camera.setText(_translate("MainWindow", "start camera"))