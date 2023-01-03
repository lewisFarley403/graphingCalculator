import switchingWidget
from simultaneousEquations import EquationViewer
from functionalityPages import Calculate, solveSystemsOf2Eq, CartGraphing, ParametricGraphing
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from texRenderWidget import TexText, TexTextWithDomain
from threading import main_thread
import calculateScreen
from PyQt5.QtWidgets import QMessageBox


# todo:
# forequal function
# display index choice
# control the domain
# then done :)

# press new to add a new equation
# type the eqution


class GraphingScreen(calculateScreen.Ui_MainWindow):
    def __init__(self, MainWindow):

        super().__init__(MainWindow)
        self.equations = []
        self.shiftClicked = False
        self.index = 0

    def setupUi(self, MainWindow):

        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.graphLayout.removeWidget(self.texRender)

        # self.domains = QtWidgets.QGridLayout(self.gridLayoutWidget_3)

        self.new.clicked.connect(self.addEquation)
        self.shift.clicked.connect(self.forShift)
        # self.xButton.clicked.connect(lambda: self.refreshDisplay('x'))

        # self.addEquation()

    def refreshDisplay(self, value, forCalc=True):

        if self.shiftClicked == True:
            print('in shift')
            if int(value) <= len(self.equations):
                self.index = int(value)-1
            else:
                self.index = -1
            print(f'index is {self.index}')
            self.shiftClicked = False

        else:
            # may need adj as self.equations changes
            self.equations[self.index].refreshCanvas(value)
            if forCalc == True:
                self.expressionForCalc += value

    def forShift(self):
        self.shiftClicked = not self.shiftClicked

    def addEquation(self):
        if self.shiftClicked == True:
            textViewer = TexTextWithDomain(self.MainWindow)
            self.shiftClicked = False
        else:
            textViewer = TexText(self.MainWindow)

        self.equations.append(textViewer)
        self.graphLayout.addWidget(textViewer)

    def forEqual(self):
        # mainPlot = CartGraphing(self.equations[-1].getEquation())
        # graphs = []
        # for eq in self.equations[:-1]:
        #     graphs.append(CartGraphing(eq.getEquation()).plotCoords())
        # mainPlot.createPlot(otherCoords=graphs)

        try:
            allCoords = []
            for eq in self.equations:
                if eq.type == 'normal':
                    p = CartGraphing(eq.getEquation())
                    print(f'the equation is {eq.getEquation()}')
                    coords = p.plotCoords()

                    allCoords.append(coords)
                    # print('graphing a normal function')
                    # print(allCoords)
                else:
                    p = ParametricGraphing(
                        eq.x.getEquation(), eq.y.getEquation())
                    p.plotCoords()
                    allCoords.append(p.coords)
            p.createPlot(otherCoords=allCoords)
        except Exception as e:
            print(e)
            msg = QMessageBox()

            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")

            msg.setInformativeText('Invalid expression')
            msg.setWindowTitle("Error")
            msg.exec_()

    def rightArrow(self):
        self.equations[self.index].switchEquation()

    def leftArrow(self):
        self.rightArrow()

    def forDel(self):
        self.equations[self.index].clearCanvas()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = GraphingScreen(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
