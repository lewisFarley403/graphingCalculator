from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate
import matplotlib.pyplot as plt


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate
import matplotlib.pyplot as plt
from typing import Optional

from utils import Stack


class TexText(QtWidgets.QWidget):
    """A widget for displaying and editing a mathematical equation.

    This widget displays a mathematical equation using a LaTeX rendering of the
    equation, and allows the user to edit the equation by adding or removing
    characters. It also provides methods for clearing the equation and setting
    the answer to the equation.
    """

    def __init__(self, parent: Optional["QWidget"] = None):
        """Initialize the TexText widget.

        Args:
            parent (QWidget): The parent widget of the TexText widget.
        """
        super().__init__()
        self.type = 'normal'  # Set text type to normal
        self.stateStack = Stack()
        self.canvasParent = parent  # Set parent widget for the canvas
        self.title = r''  # Initialize title as empty string
        self.answer = r''  # Initialize answer as empty string
        # Create a new canvas with the default empty values
        self.createNewCanvas(first=True)
        self.setUpUi()  # Set up the user interface

    def setUpUi(self):
        """Method for setting up the user interface"""
        self.layout = QtWidgets.QGridLayout()  # Create grid layout
        self.setLayout(self.layout)  # Set widget's layout to grid layout
        # Add canvas widget to the layout
        self.layout.addWidget(self.canvas, 1, 0)

    def createNewCanvas(self, first=False):
        """Method for creating a new canvas"""
        if first == False and self.title != '':
            # Create a new MplCanvas object with the given title and answer
            self.canvas = MplCanvas(
                r''+fr'${self.title}$', r''+self.answer, parent=self.canvasParent)
        else:
            # Create a new MplCanvas object with empty title and answer
            self.canvas = MplCanvas(
                r'', r'', parent=self.canvasParent)

    def refreshCanvas(self, value):
        # Concatenate the current value of the title attribute with the value parameter
        self.title += value
        self.stateStack.push(self.title)
        self.__updateDisplay()

    def __updateDisplay(self):
        # Remove the current canvas object from the layout
        self.layout.removeWidget(self.canvas)
        # Create a new canvas object using the updated title and answer attributes
        self.createNewCanvas()

        # Add the new canvas object to the layout
        self.layout.addWidget(self.canvas, 1, 0)

    def undo(self):
        print(self.stateStack.getList())
        self.stateStack.pop()
        state = self.stateStack.peek()

        if state == None:
            state = ''
        self.title = state
        self.__updateDisplay()

    def clearCanvas(self):
        self.title = r''
        self.answer = r''
        self.refreshCanvas('')

    def setAnswer(self, ans):
        self.answer = ans
        ans = str(round(float(ans), 4))
        self.refreshCanvas('')
        self.stateStack.pop()  # dont want it adding, its an answer

    def setTitle(self, val):
        self.title = val
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
            # Create a new MplCanvas object with the given title and answer
            canvas = MplCanvas(
                r''+fr'${self.title}$', r''+self.answer, parent=self.canvasParent)
        else:
            # Create a new MplCanvas object with empty title and answer
            canvas = MplCanvas(
                r'', r'', parent=self.canvasParent)
        return canvas

    def refreshCanvas(self, value):
        if value == 'x':
            value = 't'
        if self.editting == 0:
            # If the user is currently editing the x equation, update the x TexText object
            self.x.refreshCanvas(value)
        else:
            # If the user is currently editing the y equation, update the y TexText object
            self.y.refreshCanvas(value)

    def setUpUi(self):
        # Create a grid layout for the widget
        self.layout = QtWidgets.QGridLayout()
        # Set the widget's layout to the grid layout
        self.setLayout(self.layout)
        # Add a label for the x equation to the layout
        self.layout.addWidget(QtWidgets.QLabel('x: '), 1, 0)
        # Add the x TexText object to the layout
        self.layout.addWidget(self.x, 1, 1)
        # Add a label for the y equation to the layout
        self.layout.addWidget(QtWidgets.QLabel('y: '), 1, 2)
        # Add the y TexText object to the layout
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
        self.axes.text(0, 30, self.answer, fontsize=15)

        # use a lower number to make more vertical space
