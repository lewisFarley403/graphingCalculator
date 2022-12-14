import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QFont
from scipy.stats import binom
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate, BinomialDist
from texRenderWidget import TexText


class Ui_MainWindow(QtWidgets.QWidget):

    def setupUi(self, MainWindow):
        # MainWindow.resize(500, 1000)

        print('SETTING UP UI')
        self.calc = Calculate({}, {})
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.inputLayoutWrapper = QtWidgets.QWidget(self.centralwidget)
        self.resultLayoutWrapper = QtWidgets.QWidget(self.centralwidget)

        # Create input fields for number of trials, probability of success,
        # upper bound, and lower bound
        self.numTrialsLabel = QtWidgets.QLabel("Number of Trials:")
        self.numTrialsInput = QtWidgets.QLineEdit('i am here')

        self.probSuccessLabel = QtWidgets.QLabel("Probability of Success:")
        self.probSuccessInput = QtWidgets.QLineEdit()

        self.upperBoundLabel = QtWidgets.QLabel("Upper Bound:")
        self.upperBoundInput = QtWidgets.QLineEdit()

        self.lowerBoundLabel = QtWidgets.QLabel("Lower Bound:")
        self.lowerBoundInput = QtWidgets.QLineEdit()

        # Create a button to compute the cumulative distribution
        self.computeButton = QtWidgets.QPushButton(
            "Compute Cumulative Distribution")

        # Create a label and text field to display the result
        self.resultLabel = QtWidgets.QLabel("Result:")
        self.resultField = QtWidgets.QLineEdit()

        # Add the input fields and buttons to the UI
        self.inputLayout = QtWidgets.QGridLayout(self.inputLayoutWrapper)

        # Add the input fields and buttons to the UI
        self.resultLayout = QtWidgets.QGridLayout(self.resultLayoutWrapper)
        self.inputLayout.addWidget(self.numTrialsLabel, 0, 0)
        self.inputLayout.addWidget(self.numTrialsInput, 0, 1)
        self.inputLayout.addWidget(self.probSuccessLabel, 1, 0)
        self.inputLayout.addWidget(self.probSuccessInput, 1, 1)
        self.inputLayout.addWidget(self.upperBoundLabel, 2, 0)
        self.inputLayout.addWidget(self.upperBoundInput, 2, 1)
        self.inputLayout.addWidget(self.lowerBoundLabel, 3, 0)
        self.inputLayout.addWidget(self.lowerBoundInput, 3, 1)
        self.inputLayout.addWidget(self.computeButton, 4, 0)

        # Create a layout for the result label and text field
        self.resultLayout = QtWidgets.QHBoxLayout()
        self.resultLayout.addWidget(self.resultLabel)
        self.resultLayout.addWidget(self.resultField)

        # Create a main layout to hold the input and result layouts
        self.mainLayout = QtWidgets.QGridLayout()
        self.mainLayout.addWidget(self.inputLayoutWrapper)
        self.mainLayout.addWidget(self.resultLayoutWrapper)

        self.computeButton.clicked.connect(self.computeCDF)

        # Set the layout of the main window to the main layout
        # self.setLayout(self.mainLayout)

    def computeCDF(self):
        # Get the values of the input fields
        num_trials = self.numTrialsInput.text()
        prob_success = self.probSuccessInput.text()
        upper_bound = self.upperBoundInput.text()
        lower_bound = self.lowerBoundInput.text()

        # Compute the cumulative distribution
        bd = BinomialDist(num_trials, prob_success)

        result = bd.cumulative_probability(upper_bound, lower_bound)

        # Display the result in the result field
        self.resultField.setText(str(result))


if __name__ == "__main__":

    # Create a MainWindow object
    app = QApplication(sys.argv)
    # Create an instance of the Ui_MainWindow widget
    ui = Ui_MainWindow()
    # Pass the MainWindow object to the setupUi method of the Ui_MainWindow class
    MainWindow = QtWidgets.QMainWindow()

    ui.setupUi(MainWindow)

    # Show the widget
    MainWindow.show()
    sys.exit(app.exec_())
