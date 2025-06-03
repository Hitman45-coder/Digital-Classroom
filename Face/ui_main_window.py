# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QColumnView, QDialog, QGraphicsView,
    QGridLayout, QPushButton, QSizePolicy, QWidget)

class Ui_Start_2(object):
    def setupUi(self, Start_2):
        if not Start_2.objectName():
            Start_2.setObjectName(u"Start_2")
        Start_2.resize(407, 330)
        self.gridLayout = QGridLayout(Start_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.graphicsView = QGraphicsView(Start_2)
        self.graphicsView.setObjectName(u"graphicsView")

        self.gridLayout.addWidget(self.graphicsView, 0, 0, 5, 1)

        self.Start = QPushButton(Start_2)
        self.Start.setObjectName(u"Start")

        self.gridLayout.addWidget(self.Start, 0, 1, 1, 1)

        self.pushButton_2 = QPushButton(Start_2)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.gridLayout.addWidget(self.pushButton_2, 1, 1, 1, 1)

        self.pushButton_3 = QPushButton(Start_2)
        self.pushButton_3.setObjectName(u"pushButton_3")

        self.gridLayout.addWidget(self.pushButton_3, 2, 1, 1, 1)

        self.pushButton = QPushButton(Start_2)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout.addWidget(self.pushButton, 3, 1, 1, 1)

        self.columnView = QColumnView(Start_2)
        self.columnView.setObjectName(u"columnView")

        self.gridLayout.addWidget(self.columnView, 4, 1, 1, 1)


        self.retranslateUi(Start_2)

        QMetaObject.connectSlotsByName(Start_2)
    # setupUi

    def retranslateUi(self, Start_2):
        Start_2.setWindowTitle(QCoreApplication.translate("Start_2", u"Dialog", None))
        self.Start.setText(QCoreApplication.translate("Start_2", u"Start", None))
        self.pushButton_2.setText(QCoreApplication.translate("Start_2", u"Stop", None))
        self.pushButton_3.setText(QCoreApplication.translate("Start_2", u"Attendance", None))
        self.pushButton.setText(QCoreApplication.translate("Start_2", u"Mail", None))
    # retranslateUi

