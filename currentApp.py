from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate
from texRenderWidget import TexText
from calculateScreen import Ui_MainWindow
from simultaneousEquations import simScreen3Eq, SimScreen2Eq
from graphingScreen import GraphingScreen
from binomal import BinomialWidget
from normal import NormalWidget


class currentApp(object):
    def __init__(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")

        self.calc = Ui_MainWindow()
        self.calc.setupUi(MainWindow)
        self.calc.switch.clicked.connect(self.foo)

        self.sim2eq = SimScreen2Eq()
        self.sim3eq = simScreen3Eq()
        self.binomial = BinomialWidget()
        self.normal = NormalWidget()


# 114
        # self.sim3eq.setupUi(MainWindow)
        self.switch = QtWidgets.QPushButton(MainWindow)
        self.graphing = GraphingScreen(MainWindow)
        self.currentIndex = 1
        self.switch.clicked.connect(self.foo)
        self.screens = [self.calc, self.sim2eq,
                        self.sim3eq, self.graphing, self.binomial, self.normal]

    def foo(self):
        # do this for all pages
        print('calling', self.currentIndex)
        # self.sim3eq.setupUi(self.MainWindow)
        currentPage = self.screens[self.currentIndex]
        currentPage.setupUi(self.MainWindow)
        currentPage.switch.clicked.connect(self.foo)

        self.currentIndex += 1
        self.currentIndex %= len(self.screens)
        self.switch = QtWidgets.QPushButton(MainWindow)
        self.switch.clicked.connect(self.foo)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    # ui = SimScreen2Eq()

    ui = currentApp(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
