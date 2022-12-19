import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMainWindow
from PyQt5.QtGui import QFont
from scipy.stats import binom
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functionalityPages import Calculate, BinomialDist, NormalDist


import sys
from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtWidgets import QMessageBox


class NormalWidget(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        # create the input fields

        self.meanInput = QtWidgets.QLineEdit()
        self.meanLabel = QtWidgets.QLabel('Mean:')
        self.stdvInput = QtWidgets.QLineEdit()
        self.stdvLabel = QtWidgets.QLabel('Standard Deviation:')
        self.lower_bound_input = QtWidgets.QLineEdit()
        self.lower_bound_label = QtWidgets.QLabel('Lower Bound:')
        self.upper_bound_input = QtWidgets.QLineEdit()
        self.upper_bound_label = QtWidgets.QLabel('Upper Bound:')

        # create the execute button
        self.execute_button = QtWidgets.QPushButton('Execute')
        self.switch = QtWidgets.QPushButton('switch')
        self.answer = QtWidgets.QLabel('Upper Bound:')

        # create a layout to hold the input fields and button
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.addWidget(self.meanLabel, 0, 0)
        self.gridLayout.addWidget(self.meanInput, 0, 1)
        self.gridLayout.addWidget(self.stdvLabel, 1, 0)
        self.gridLayout.addWidget(self.stdvInput, 1, 1)
        self.gridLayout.addWidget(self.lower_bound_label, 2, 0)
        self.gridLayout.addWidget(self.lower_bound_input, 2, 1)
        self.gridLayout.addWidget(self.upper_bound_label, 3, 0)
        self.gridLayout.addWidget(self.upper_bound_input, 3, 1)
        self.gridLayout.addWidget(self.execute_button, 4, 0, 1, 2)
        self.gridLayout.addWidget(self.answer, 5, 0)

        self.gridLayout.addWidget(self.switch, 6, 0, 1, 2)

        # connect the execute button to a function
        self.execute_button.clicked.connect(self.execute)

        # set the central widget for the main window
        MainWindow.setCentralWidget(self)

    def execute(self):

        # get the values from the input fields
        errors = []

        n = int(self.meanInput.text())

        p = float(self.stdvInput.text())
        if p <= 0:
            errors.append('invalid standard devation ')

        lower_bound = int(self.lower_bound_input.text())
        upper_bound = int(self.upper_bound_input.text())

        if len(errors) != 0:
            # error
            msg = QMessageBox()

            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(f'{"".join(errors)}')
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            sys.setrecursionlimit((n+1)*(lower_bound+1)*(upper_bound+1))
            print(sys.getrecursionlimit())

            distribution = NormalDist(n, p)
            probability = distribution.cumulative_probability(
                lower_bound, upper_bound)
            self.answer.setText(f'p: {str(probability)}')
            # do something with the values (e.g., print them)
            print(
                f'n: {n}, p: {p}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = BinomialWidget()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())
