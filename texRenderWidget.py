from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate
import matplotlib.pyplot as plt


class TexText(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.type = 'normal'

        self.canvasParent = parent
        self.title = r''
        self.answer = r''
        self.createNewCanvas(first=True)
        self.setUpUi()

    def setUpUi(self):
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.canvas, 1, 0)

    def createNewCanvas(self, first=False):
        if first == False and self.title != '':
            self.canvas = MplCanvas(
                r''+fr'${self.title}$', r''+self.answer, parent=self.canvasParent)
        else:

            self.canvas = MplCanvas(
                r'', r'', parent=self.canvasParent)

    def refreshCanvas(self, value):
        self.title += value
        self.layout.removeWidget(self.canvas)
        self.createNewCanvas()

        # self.graph = MplCanvas(fr'${self.title}$', parent=self.layout)

        self.layout.addWidget(self.canvas, 1, 0)

    def clearCanvas(self):
        self.title = r''
        self.answer = r''
        self.refreshCanvas('')

    def setAnswer(self, ans):
        self.answer = ans
        self.refreshCanvas('')

    def getEquation(self):
        return self.title

    def switchEquation(self):
        pass


class TexTextWithDomain(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.type = 'domain'

        self.canvasParent = parent
        # self.x = self.createNewCanvas(first=True)
        # self.y = self.createNewCanvas(first=True)
        self.x = TexText()
        self.y = TexText()
        self.setUpUi()
        self.editting = 0

    #     self.upper = ''
    #     self.lower = ''
    #     # test adding a label
    #     self.canvases = [self.canvas(),self.canvas()]

    def createNewCanvas(self, first=False):
        if first == False and self.title != '':
            canvas = MplCanvas(
                r''+fr'${self.title}$', r''+self.answer, parent=self.canvasParent)
        else:

            canvas = MplCanvas(
                r'', r'', parent=self.canvasParent)
        return canvas

    def refreshCanvas(self, value):
        if value == 'x':
            value = 't'
        if self.editting == 0:
            self.x.refreshCanvas(value)
        else:
            self.y.refreshCanvas(value)

    def setUpUi(self):
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        # self.testLabel = QtWidgets.QLabel('x: ')
        # self.layout.addWidget(self.testLabel, 1, 1)
        # self.layout.addWidget(self.canvas, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel('x: '), 1, 0)
        self.layout.addWidget(self.x, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel('y: '), 1, 2)

        self.layout.addWidget(self.y, 1, 3)

    def switchEquation(self):
        if self.editting == 0:
            self.editting = 1
        else:
            self.editting = 0

    def clearCanvas(self):
        if self.editting == 0:
            self.x.title = r''

            self.x.refreshCanvas('')
        else:
            self.y.title-r''
            self.y.refreshCanvas('')


class MplCanvas(FigureCanvas):
    def __init__(self, title, answer, parent=None, dpi=100):
        self.answer = r''+answer
        fig = Figure(dpi=dpi)
        fig.subplots_adjust(bottom=0.01, top=0.02)
        super().__init__(fig)
        self.axes = fig.add_subplot(111)
        self.title = title
        # self.axes.set_title(self.title, pad=-100)
        self.axes.text(0, 50, self.title, fontsize=20)
        self.axes.text(1, 50, self.answer, fontsize=20)

        # use a lower number to make more vertical space
