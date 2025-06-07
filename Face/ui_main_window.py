# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main_window.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QHeaderView, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QWidget)

class Ui_Start_2(object):
    def setupUi(self, Start_2):
        if not Start_2.objectName():
            Start_2.setObjectName(u"Start_2")
        Start_2.resize(1129, 716)
        self.Stop_Button = QPushButton(Start_2)
        self.Stop_Button.setObjectName(u"Stop_Button")
        self.Stop_Button.setGeometry(QRect(630, 80, 421, 29))
        self.Add_Button = QPushButton(Start_2)
        self.Add_Button.setObjectName(u"Add_Button")
        self.Add_Button.setGeometry(QRect(630, 180, 421, 29))
        self.Mail_Button = QPushButton(Start_2)
        self.Mail_Button.setObjectName(u"Mail_Button")
        self.Mail_Button.setGeometry(QRect(630, 130, 421, 29))
        self.Remove_Button = QPushButton(Start_2)
        self.Remove_Button.setObjectName(u"Remove_Button")
        self.Remove_Button.setGeometry(QRect(630, 230, 421, 29))
        self.video_container = QWidget(Start_2)
        self.video_container.setObjectName(u"video_container")
        self.video_container.setEnabled(True)
        self.video_container.setGeometry(QRect(15, 20, 601, 629))
        self.Start_Button = QPushButton(Start_2)
        self.Start_Button.setObjectName(u"Start_Button")
        self.Start_Button.setGeometry(QRect(630, 30, 421, 29))
        self.Attendance = QTableWidget(Start_2)
        self.Attendance.setObjectName(u"Attendance")
        self.Attendance.setGeometry(QRect(615, 271, 461, 381))

        self.retranslateUi(Start_2)

        QMetaObject.connectSlotsByName(Start_2)
    # setupUi

    def retranslateUi(self, Start_2):
        Start_2.setWindowTitle(QCoreApplication.translate("Start_2", u"Dialog", None))
        self.Stop_Button.setText(QCoreApplication.translate("Start_2", u"Stop", None))
        self.Add_Button.setText(QCoreApplication.translate("Start_2", u"Add Face", None))
        self.Mail_Button.setText(QCoreApplication.translate("Start_2", u"Mail", None))
        self.Remove_Button.setText(QCoreApplication.translate("Start_2", u"Remove Face", None))
        self.Start_Button.setText(QCoreApplication.translate("Start_2", u"Start", None))
    # retranslateUi

