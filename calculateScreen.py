# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui2.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate
from texRenderWidget import TexText
# import switchingWidget

WIDTH = 800
HEIGHT = 800
SF = 1.5


class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        print('SETTING UP UI')
        self.calc = Calculate({}, {})
        MainWindow.setObjectName("MainWindow")
        self.flags = {'sqrt': False, 'fractDenom': False, 'fractNum': False}
        self.texEq = r''
        self.expressionForCalc = ''

        MainWindow.resize(1000, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralGrid = QtWidgets.QGridLayout(self.centralwidget)
        self.graphLayoutWrapper = QtWidgets.QWidget(self.centralwidget)
        self.grid2Wrapper = QtWidgets.QWidget(self.centralwidget)

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        # self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 700, 1000, 141))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        # self.switchingWidget = switchingWidget.switchingWidget({}, MainWindow)
        # self.gridLayout.addWidget(self.switchingWidget)
        self.no8 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no8.setObjectName("no8")
        self.gridLayout.addWidget(self.no8, 0, 1, 1, 1)
        self.sub = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no8.clicked.connect(lambda: self.refreshDisplay('8'))

        self.sub.setObjectName("sub")
        self.sub.clicked.connect(lambda: self.refreshDisplay('-'))
        self.gridLayout.addWidget(self.sub, 2, 4, 1, 1)
        self.no6 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no6.clicked.connect(lambda: self.refreshDisplay('6'))

        self.no6.setObjectName("no6")
        self.gridLayout.addWidget(self.no6, 1, 2, 1, 1)
        self.mult = QtWidgets.QPushButton(self.gridLayoutWidget)

        self.mult.setObjectName("mult")
        self.gridLayout.addWidget(self.mult, 1, 3, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 0, 3, 1, 1)
        self.mult.clicked.connect(lambda: self.refreshDisplay('*'))

        self.no1 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no1.setObjectName("no1")
        self.no1.clicked.connect(lambda: self.refreshDisplay('1'))
        self.gridLayout.addWidget(self.no1, 2, 0, 1, 1)

        self.no2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no2.setObjectName("no2")
        self.gridLayout.addWidget(self.no2, 2, 1, 1, 1)
        self.no2.clicked.connect(lambda: self.refreshDisplay('2'))

        self.no7 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no7.setObjectName("no7")
        self.gridLayout.addWidget(self.no7, 0, 0, 1, 1)
        self.no7.clicked.connect(lambda: self.refreshDisplay('7'))

        self.no9 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no9.setObjectName("no9")
        self.gridLayout.addWidget(self.no9, 0, 2, 1, 1)
        self.no9.clicked.connect(lambda: self.refreshDisplay('9'))

        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 4, 1, 1)

        self.no3 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no3.setObjectName("no3")
        self.gridLayout.addWidget(self.no3, 2, 2, 1, 1)
        self.no3.clicked.connect(lambda: self.refreshDisplay('3'))

        self.add = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.add.setObjectName("add")
        self.gridLayout.addWidget(self.add, 2, 3, 1, 1)
        self.add.clicked.connect(lambda: self.refreshDisplay('+'))

        self.no4 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no4.setObjectName("no4")
        self.gridLayout.addWidget(self.no4, 1, 0, 1, 1)
        self.no4.clicked.connect(lambda: self.refreshDisplay('4'))

        self.no5 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no5.setObjectName("no5")
        self.gridLayout.addWidget(self.no5, 1, 1, 1, 1)
        self.no5.clicked.connect(lambda: self.refreshDisplay('5'))

        self.div = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.div.setObjectName("div")
        self.gridLayout.addWidget(self.div, 1, 4, 1, 1)
        self.div.clicked.connect(lambda: self.refreshDisplay('/'))

        self.no0 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.no0.setObjectName("no0")
        self.gridLayout.addWidget(self.no0, 3, 0, 1, 1)
        self.no0.clicked.connect(lambda: self.refreshDisplay('0'))

        self.decimal = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.decimal.setObjectName("decimal")
        self.decimal.clicked.connect(lambda: self.refreshDisplay('.'))
        self.gridLayout.addWidget(self.decimal, 3, 1, 1, 1)
        self.pushButton_18 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_18.setObjectName("pushButton_18")
        self.gridLayout.addWidget(self.pushButton_18, 3, 2, 1, 1)
        self.ans = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.ans.setObjectName("ans")
        self.gridLayout.addWidget(self.ans, 3, 3, 1, 1)
        self.equal = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.equal.setObjectName("equal")
        self.equal.clicked.connect(self.forEqual)
        self.gridLayout.addWidget(self.equal, 3, 4, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        # self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 700, 1000, 166))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_40 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_40.setObjectName("pushButton_40")
        self.gridLayout_2.addWidget(self.pushButton_40, 3, 1, 1, 1)

        self.square = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.square.setObjectName("square")
        self.gridLayout_2.addWidget(self.square, 1, 2, 1, 1)
        self.square.clicked.connect(lambda: self.refreshDisplay('^2'))

        self.logbx = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.logbx.setObjectName("logbx")
        self.gridLayout_2.addWidget(self.logbx, 1, 4, 1, 1)
        self.pi = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pi.setObjectName("pi")
        self.pi.clicked.connect(lambda: self.refreshDisplay('π'))
        self.gridLayout_2.addWidget(self.pi, 2, 2, 1, 1)

        self.fract = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.fract.setObjectName("fract")
        self.gridLayout_2.addWidget(self.fract, 1, 0, 1, 1)
        self.fract.clicked.connect(self.fracButton)
        self.pushButton_25 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_25.setObjectName("pushButton_25")
        self.gridLayout_2.addWidget(self.pushButton_25, 0, 1, 1, 1)
        self.pow = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pow.setObjectName("pow")
        self.gridLayout_2.addWidget(self.pow, 1, 3, 1, 1)
        self.shift = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.shift.setObjectName("shift")
        self.gridLayout_2.addWidget(self.shift, 0, 0, 1, 1)
        self.tan = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.tan.setObjectName("tan")
        self.gridLayout_2.addWidget(self.tan, 2, 5, 1, 1)
        self.tan.clicked.connect(lambda: self.refreshDisplay('tan'))

        self.delete = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.delete.setObjectName("delete")
        self.gridLayout_2.addWidget(self.delete, 3, 5, 1, 1)
        self.delete.clicked.connect(self.forDel)
        self.switch = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.switch.setObjectName("switch")
        self.gridLayout_2.addWidget(self.switch, 0, 5, 1, 1)
        self.rightBracket = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.rightBracket.setObjectName(")")
        self.rightBracket.clicked.connect(lambda: self.refreshDisplay(')'))
        self.gridLayout_2.addWidget(self.rightBracket, 3, 3, 1, 1)
        self.xButton = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.xButton.setObjectName("xButton")
        self.gridLayout_2.addWidget(self.xButton, 3, 0, 1, 1)
        self.pushButton_32 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_32.setObjectName("pushButton_32")
        self.gridLayout_2.addWidget(self.pushButton_32, 1, 5, 1, 1)
        self.cos = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.cos.setObjectName("cos")
        self.gridLayout_2.addWidget(self.cos, 2, 4, 1, 1)
        self.cos.clicked.connect(lambda: self.refreshDisplay('cos('))

        self.pushButton_33 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_33.setObjectName("pushButton_33")
        self.gridLayout_2.addWidget(self.pushButton_33, 2, 0, 1, 1)
        self.pushButton_34 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_34.setObjectName("pushButton_34")
        self.gridLayout_2.addWidget(self.pushButton_34, 2, 1, 1, 1)
        self.new = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.new.setObjectName("new")
        self.gridLayout_2.addWidget(self.new, 0, 4, 1, 1)
        self.sin = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.sin.setObjectName("sin")
        self.gridLayout_2.addWidget(self.sin, 2, 3, 1, 1)
        self.sin.clicked.connect(lambda: self.refreshDisplay('sin('))

        self.sqrt = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.sqrt.setObjectName("sqrt")
        self.gridLayout_2.addWidget(self.sqrt, 1, 1, 1, 1)
        self.pushButton_43 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_43.setObjectName("pushButton_43")
        self.gridLayout_2.addWidget(self.pushButton_43, 3, 4, 1, 1)
        self.leftBracket = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.leftBracket.setObjectName("leftBracket")
        self.leftBracket.clicked.connect(lambda: self.refreshDisplay('('))
        self.gridLayout_2.addWidget(self.leftBracket, 3, 2, 1, 1)
        self.pushButton_23 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_23.clicked.connect(self.rightArrow)
        self.gridLayout_2.addWidget(self.pushButton_23, 0, 3, 1, 1)
        self.pushButton_24 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_24.setObjectName("pushButton_24")
        self.pushButton_24.clicked.connect(self.leftArrow)
        self.gridLayout_2.addWidget(self.pushButton_24, 0, 2, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 801, 231))  # origional
        # self.widget.setGeometry(QtCore.QRect(0, 0, WIDTH, HEIGHT))

        self.widget.setObjectName("widget")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.widget)
        self.gridLayoutWidget_3.setGeometry(
            QtCore.QRect(0, 0, 801, 231))  # origional
        # self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 0, WIDTH, HEIGHT))

        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.graphLayout = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.graphLayout.setContentsMargins(0, 0, 0, 0)
        self.graphLayout.setObjectName("graphLayout")

        # playing with tex text edit
        self.texRender = TexText(parent=self.graphLayout)

        # self.graph = MplCanvas(r'', parent=self.gridLayout)  # og

        # self.graphLayout.addWidget(self.graph) og
        self.graphLayout.addWidget(self.texRender)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.centralGrid.addWidget(self.gridLayoutWidget_3)

        # self.centralGrid.addWidget(self.widget)

        self.centralGrid.addWidget(self.gridLayoutWidget_2)
        self.centralGrid.addWidget(self.gridLayoutWidget)
        # MainWindow.setStyleSheet(open("style.qss", "r").read())

    # def refreshDisplay(self, value, forCalc=True):

    #     self.texEq += value
    #     if forCalc == True:
    #         self.expressionForCalc += value
    #     print(self.expressionForCalc)
    #     self.graphLayout.removeWidget(self.graph)
    #     self.graph = MplCanvas(fr'${self.texEq}$', parent=self.gridLayout)
    #     self.graphLayout.addWidget(self.graph)
    def leftArrow(self):
        pass

    def forDel(self):
        # overide when another screen inherits and does whatever it wants with the delete function
        self.texEq = r''
        print('in for del')
        self.expressionForCalc = ''
        self.texRender.clearCanvas()
        self.refreshDisplay('')

    def refreshDisplay(self, value, forCalc=True):
        self.texRender.refreshCanvas(value)
        if forCalc == True:
            self.expressionForCalc += value

    def rightArrow(self):
        if self.flags['fractNum'] == True:
            self.refreshDisplay('}{', forCalc=False)
            self.expressionForCalc += ')/('
            self.flags['fractNum'] = False
            self.flags['fractDenom'] = True
        elif self.flags['fractDenom'] == True:
            self.expressionForCalc += ')'
            self.flags['fractDenom'] = False
            self.refreshDisplay('}', forCalc=False)

    def fracButton(self):
        # example fraction syntax r'$\frac{9}{3}$'
        print('fract button')
        self.refreshDisplay(r'\frac{', forCalc=False)
        self.expressionForCalc += '('
        self.flags['fractNum'] = True

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.no8.setText(_translate("MainWindow", "8"))
        self.sub.setText(_translate("MainWindow", "-"))

        self.no6.setText(_translate("MainWindow", "6"))
        self.mult.setText(_translate("MainWindow", "X"))
        self.pushButton_5.setText(_translate("MainWindow", "DEL"))
        self.no1.setText(_translate("MainWindow", "1"))
        self.no2.setText(_translate("MainWindow", "2"))
        self.no7.setText(_translate("MainWindow", "7"))
        self.no9.setText(_translate("MainWindow", "9"))
        self.pushButton_4.setText(_translate("MainWindow", "AC"))
        self.no3.setText(_translate("MainWindow", "3"))
        self.add.setText(_translate("MainWindow", "+"))
        self.no4.setText(_translate("MainWindow", "4"))
        self.no5.setText(_translate("MainWindow", "5"))
        self.div.setText(_translate("MainWindow", "DIV"))
        self.no0.setText(_translate("MainWindow", "0"))
        self.decimal.setText(_translate("MainWindow", "."))
        self.pushButton_18.setText(_translate("MainWindow", "x10"))
        self.ans.setText(_translate("MainWindow", "Ans"))
        self.equal.setText(_translate("MainWindow", "="))
        self.pushButton_40.setText(_translate("MainWindow", "pb1"))
        self.square.setText(_translate("MainWindow", "^2"))
        self.logbx.setText(_translate("MainWindow", "logb x"))
        self.pi.setText(_translate("MainWindow", "π"))
        self.fract.setText(_translate("MainWindow", "Fract"))
        self.pushButton_25.setText(_translate("MainWindow", "DOWN"))
        self.pow.setText(_translate("MainWindow", "^n"))
        self.shift.setText(_translate("MainWindow", "Shift"))
        self.tan.setText(_translate("MainWindow", "tan"))
        self.delete.setText(_translate("MainWindow", "DEL"))
        self.switch.setText(_translate("MainWindow", "switch"))
        self.rightBracket.setText(_translate("MainWindow", ")"))
        self.xButton.setText(_translate("MainWindow", "X"))
        self.pushButton_32.setText(_translate("MainWindow", "ln"))
        self.cos.setText(_translate("MainWindow", "cos"))
        self.pushButton_33.setText(_translate("MainWindow", "-"))
        self.pushButton_34.setText(_translate("MainWindow", "PushButton"))
        self.new.setText(_translate("MainWindow", "New"))
        self.sin.setText(_translate("MainWindow", "sin"))
        self.sqrt.setText(_translate("MainWindow", "sqrt"))
        self.pushButton_43.setText(_translate("MainWindow", "pushButton"))
        self.leftBracket.setText(_translate("MainWindow", "("))
        self.pushButton_23.setText(_translate("MainWindow", "-->"))
        self.pushButton_24.setText(_translate("MainWindow", "<--"))

    def forEqual(self):
        print(f'COMPUTING {self.expressionForCalc}')
        result = self.calc.computeExpression(self.expressionForCalc)
        self.texRender.setAnswer(str(result))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())