import calculateScreen
from texRenderWidget import TexText

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate, solveSystemsOf2Eq, solveSystemOf3Eq
from texRenderWidget import TexText


class EquationViewer(TexText):
    def __init__(self, numUnknowns, parent=None):
        super().__init__(parent=parent)
        self.numUnknowns = numUnknowns
        # plus one for the answer
        self.unknowns = ['' for i in range(numUnknowns+1)]

        self.changeUnknown(0, '')

    def changeUnknown(self, index, value):
        self.unknowns[index] += str(value)
        self.clearCanvas()
        self.refreshCanvas(self.getStringRep())

    def getStringRep(self):
        # x,y for 2
        # xyz for 3
        if self.numUnknowns == 3:
            return f'{self.unknowns[0]} x + {self.unknowns[1]} y + {self.unknowns[2]} z = {self.unknowns[3]}'
        return f'{self.unknowns[0]} x + {self.unknowns[1]} y = {self.unknowns[2]}'


class SimScreen2Eq(calculateScreen.Ui_MainWindow):
    def __init__(self):
        super().__init__()

        self.values = [[], []]  # each array for each equation
        self.selected = False
        self.eq1 = []
        self.eq2 = []
        self.currentIndex = 0

    def forEqual(self):
        if self.currentIndex >= 5:  # if we are on the last answer:
            # calculate

            #print('would calc here')
            eq1 = self.eq1View.unknowns[0:2]
            eq1 = [float(i) for i in eq1]

            eq2 = self.eq2View.unknowns[0:2]
            eq2 = [float(i) for i in eq2]
            ans1 = float(self.eq1View.unknowns[-1])
            ans2 = float(self.eq2View.unknowns[-1])
            print(eq1, eq2, ans1, ans2)

            # solveSystemsOf2Eq([self.eq1View.unknowns[0:1], self.eq2View.unknowns[0:1]], [
            #                   [self.eq1View.unknowns[2]], self.eq2View.unknowns[2]])
            res = solveSystemsOf2Eq([eq1, eq2], [ans1, ans2])
            x = res[0][0]
            y = res[1][0]
            print(x, y)
            # set answers on screen here

        else:
            self.currentIndex += 1

    def setupUi(self, MainWindow, numUnknowns=2):
        # calling the parents class, this keeps all the functionality and allows me to keep the method names consistent, but still modify this method
        super().setupUi(MainWindow)
        self.eq1View = EquationViewer(numUnknowns)

        self.graphLayout.addWidget(self.eq1View)
        self.eq2View = EquationViewer(numUnknowns)
        self.graphLayout.addWidget(self.eq2View)

    def refreshDisplay(self, value, forCalc=True):

        if self.currentIndex <= 2:
            self.eq1View.changeUnknown(self.currentIndex, value)
            pass

        else:
            self.eq2View.changeUnknown(self.currentIndex-3, value)

            # edit the second equation


class simScreen3Eq(SimScreen2Eq):
    def __init__(self):
        super().__init__()
        self.values = [[], [], []]  # each array for each equation
        self.eq3 = []
        # solveSystemOf3Eq

    def forEqual(self):
        if self.currentIndex < 11:
            self.currentIndex += 1
        else:
            print(self.values)
            eq1 = self.eq1View.unknowns[0:3]
            eq1 = [float(i) for i in eq1]

            eq2 = self.eq2View.unknowns[0:3]
            eq2 = [float(i) for i in eq2]
            eq3 = self.eq3View.unknowns[0:3]
            eq3 = [float(i) for i in eq3]
            ans1 = float(self.eq1View.unknowns[-1])
            ans2 = float(self.eq2View.unknowns[-1])
            ans3 = float(self.eq3View.unknowns[-1])
            x, y, z = solveSystemOf3Eq([eq1, eq2, eq3], [ans1, ans2, ans3])
            print(x, y, z)
            # set answers on screen here

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow, numUnknowns=3)
        self.eq3View = EquationViewer(3)
        self.graphLayout.addWidget(self.eq3View)
        self.graphLayout.removeWidget(self.texRender)

    def refreshDisplay(self, value, forCalc=True):

        if self.currentIndex <= 3:
            self.eq1View.changeUnknown(self.currentIndex, value)

        elif self.currentIndex <= 7:
            self.eq2View.changeUnknown(self.currentIndex-8, value)

            # edit the second equation
        else:
            self.eq3View.changeUnknown(self.currentIndex-8, value)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    # ui = SimScreen2Eq()

    ui = simScreen3Eq()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
