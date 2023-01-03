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
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton
from PyQt5 import Qt


class currentApp(object):
    def __init__(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")

        self.calc = Ui_MainWindow()
        self.calc.setupUi(MainWindow)
        self.calc.switch.clicked.connect(self.switchFunc)

        self.sim2eq = SimScreen2Eq()
        self.sim3eq = simScreen3Eq()
        self.binomial = BinomialWidget()
        self.normal = NormalWidget()


# 114
        # self.sim3eq.setupUi(MainWindow)
        self.switch = QtWidgets.QPushButton(MainWindow)
        self.graphing = GraphingScreen(MainWindow)
        self.currentIndex = 1
        self.screens = [self.calc, self.sim2eq,
                        self.sim3eq, self.graphing, self.binomial, self.normal]
        self.switch.clicked.connect(self.switchFunc)

    def setIndex(self, index):
        self.currentIndex = index
        self.foo()

    def switchFunc(self):

        msg = QMessageBox()
        msg.setWindowTitle("Message Box with Custom Buttons")
        msg.setText("This is a message box with custom buttons.")

        # Create custom buttons
        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(lambda: self.setIndex(0))
        simEq_2_button = QPushButton("2-variable Simultaneous Equations")
        simEq_2_button.clicked.connect(lambda: self.setIndex(1))

        simEq_3_button = QPushButton("3-variable Simultaneous Equations")
        simEq_3_button.clicked.connect(lambda: self.setIndex(2))

        normal_dist_button = QPushButton("Normal Distribution")
        normal_dist_button.clicked.connect(lambda: self.setIndex(3))

        binomial_dist_button = QPushButton("Binomial Distribution")
        binomial_dist_button.clicked.connect(lambda: self.setIndex(4))

        # Add custom buttons to the message box
        msg.addButton(calculate_button, QMessageBox.AcceptRole)
        msg.addButton(simEq_2_button, QMessageBox.AcceptRole)
        msg.addButton(simEq_3_button, QMessageBox.AcceptRole)
        msg.addButton(normal_dist_button, QMessageBox.AcceptRole)
        msg.addButton(binomial_dist_button, QMessageBox.AcceptRole)

        # Set the layout direction to be vertical

        ret = msg.exec_()

    def foo(self):
            # do this for all pages
        print('calling', self.currentIndex)
        # self.sim3eq.setupUi(self.MainWindow)
        currentPage = self.screens[self.currentIndex]
        currentPage.setupUi(self.MainWindow)
        currentPage.switch.clicked.connect(self.switchFunc)

        # self.currentIndex += 1
        # self.currentIndex %= len(self.screens)
        self.switch = QtWidgets.QPushButton(MainWindow)
        self.switch.clicked.connect(self.switchFunc)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    # ui = SimScreen2Eq()

    ui = currentApp(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
