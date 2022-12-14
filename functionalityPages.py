from rpn import RPN
from config import precedence
from utils import Stack
import utils
import math
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, List


class Calculate:
    def __init__(self, config: dict, vars: Optional[dict] = None):
        """Initialize the Calculate utility class.

        Args:
            config (dict): A dictionary of configuration settings.
            vars (dict): A dictionary of variables and their values.
        """
        if vars == None:
            self.vars = {}  # letters that have a value attatched to them
        else:
            self.vars = vars
        self.vars['π'] = math.pi

    def computeExpression(self, expression: str) -> float:
        """Compute the value of an infix mathematical expression.

        Args:
            expression (str): The infix mathematical expression to compute.

        Returns:
            The value of the computed expression.
        """
        r = RPN(expression, self.vars)  # convert infix to rpn

        return self.computeRPN(r)  # computes the value of the expression

    def computeRPN(self, RPN) -> float:
        """Compute the value of a reverse Polish notation mathematical expression.

        Args:
            RPN (RPN): The reverse Polish notation expression to compute.

        Returns:
            The value of the computed expression.
        """
        paramStack = Stack()
        # print(f'computing {RPN.rpn}')
        # print(f'rpn stack: {RPN.rpn}')
        for item in RPN.rpn:  # goes through each important part of the rpn

            if item not in precedence:  # if it is a number
                paramStack.push(item)
            else:

                # need to computer whatever operator is in here
                if item != 'sin' and item != 'cos' and item != 'tan':
                    x = paramStack.pop()
                    y = paramStack.pop()
                    # print(f'x: {x}, y: {y}')
                    res = self.computeOp(x, y, item)
                    paramStack.push(res)
                else:
                    x = paramStack.pop()
                    res = self.computeTrig(item, x)
                    paramStack.push(res)

        return paramStack.pop()  # this is the final value

    def findInput(self, op1: str) -> float:
        """Find the value of a variable or number.

        Args:
            op1 (str): The variable or number to find the value of.

        Returns:
            The value of the variable or number.
        """
        try:
            op1 = float(op1)
        except Exception:
            if op1 in self.vars:
                op1 = float(self.vars[op1])
            elif op1.find('x') != -1:
                i = op1.find('x')
                coef = float(op1[:i])
                x = self.findInput('x')
                op1 = x*coef
            else:
                op1 = None  # error, the var isnt defined
        return op1

    def computeOp(self, op1: str, op2: str, symb: str) -> float:
        """Compute the value of a mathematical operator applied to two operands.

        Args:
            op1 (str): The first operand.
            op2 (str): The second operand.
            symb (str): The mathematical operator to apply.

        Returns:
            The value of the computed operation.
        """
        op1 = self.findInput(op1)

        op2 = self.findInput(op2)

        # not compatible in python 3.6
        # match symb:
        #     case '+':
        #         return op1+op2
        #     case '*':
        #         return op1*op2
        #     case '-':
        #         return op2-op1
        #     case '/':
        #         return op2/op1
        # needs more for the functions

        if symb == '+':
            return op1+op2

        elif symb == '*':
            return op1*op2
        elif symb == '-':
            return op2-op1
        elif symb == '/':
            return op2/op1
        elif symb == '^':
            return op2**op1

    def computeTrig(self, func: str, x: str) -> float:
        """Compute the value of a trigonometric function applied to an operand.

        Args:
            func (str): The trigonometric function to apply.
            x (str): The operand to apply the function to.

        Returns:
            The value of the computed operation.
        """
        x = self.findInput(x)
        if func == 'sin':
            # return math.sin(float(x))
            return round(self.sin(float(x)), 4)
        elif func == 'cos':
            # return math.cos(float(x))
            return round(self.cos(float(x)), 4)
        else:
            return round(self.tan(float(x)), 4)

    def retrieveVariable(self, var):
        val = utils.readSetting(var)
        return val

    def stringFactorial(self, x):
        s = ''
        for i in range(1, x+1):
            s += f'{i}*'
        return s[:-1]

    def trigBuilder(self, x, nTerm):
        x %= math.pi/2
        calc = Calculate({}, vars={'x': x})
        string = ''
        for i in range(100):
            n = nTerm(i)
            # print(n)
            sign = (-1) ** i
            fact = self.stringFactorial(n)
            if sign > 0 and len(string) != 0:
                string += f'+(1/({fact}))x^{n}'
            elif sign < 0 and len(string) != 0:
                string += f'-(1/({fact}))x^{n}'
            else:
                string += 'x'
        return calc.computeExpression(string)
        # return calc.computeExpression('x-(1/(3*2))x^3+(1/(5*4*3*2))x^5')

    def sin(self, x: float)->float:
        """Compute the sine of an angle.

        Args:
            x (float): The angle in radians.

        Returns:
            The sine of the angle.
        """
        x = x % (2 * math.pi)  # Reduce the angle to the range (-2*pi, 2*pi)
        n = 15  # Number of terms in the series
        sin_x = 0
        for i in range(n):
            sin_x += (-1)**i * x**(2*i+1) / math.factorial(2*i+1)
        # print(sin_x)
        return sin_x

    def cos(self, x: float)->float:
        """Compute the cosine of an angle.

        Args:
            x (float): The angle in radians.

        Returns:
            The cosine of the angle.
        """
        x = x % (2 * math.pi)  # Reduce the angle to the range (-2*pi, 2*pi)
        n = 15  # Number of terms in the series
        cos_x = 0
        for i in range(n):
            cos_x += (-1)**i * x**(2*i) / math.factorial(2*i)
        # print(cos_x)
        return cos_x

    def tan(self, x: float)->float:
        """Compute the tan of an angle.

        Args:
            x (float): The angle in radians.

        Returns:
            The tan of the angle.
        """
        sin_x = self.sin(x)
        cos_x = self.cos(x)
        return sin_x/cos_x


'''In this code, the Calculate class is defined to perform mathematical calculations on expressions. The __init__ function initializes an instance of the Calculate class and creates an empty dictionary to store variables and their values. The computeExpression and computeRPN functions take an input expression and compute its value using Reverse Polish Notation (RPN) representation. The computeOp and computeTrig functions are used to perform mathematical operations and trigonometric functions, respectively. The findInput function retrieves the value of a variable from the dictionary of stored variables if it exists, and the retrieveVariable function is used to retrieve values for specific variables. The stringFactorial function returns a string representation of a factorial operation.

The Stack class from the utils module is used to implement a stack data structure, which is a linear data structure that allows for the insertion and removal of elements in a last-in, first-out (LIFO) manner. This data structure is used to store and manipulate the operands and operators in the RPN representation of the input expression. The precedence dictionary from the config module is used to define the order of operations for the mathematical calculations performed by the Calculate class. The RPN class from the rpn module is used to convert infix expressions to RPN representation.

These data structures were chosen to support the operations performed by the Calculate class, including the conversion of expressions to RPN representation and the evaluation of RPN expressions using a stack data structure. The Stack class and the precedence dictionary are used specifically to support the evaluation of RPN expressions, while the RPN class is used to convert infix expressions to RPN representation for easier evaluation.'''


class CartGraphing:
    def __init__(self, function: str):
        """Initialize the `CartGraphing` class.

        Args:
            function (str): The mathematical function to be plotted.
        """
        self.function = function
        self.coords = []  # xy coords
        # self.domain = [-100, 100]
        # self.domain = [-250, 250]  # the values x or t take
        self.domain = [-100, 100]  # the values x or t take
        self.dp = 3  # number of decimal places to round x to
        self.step = 1e-3  # step size for incrementing x

    def plotCoords(self) -> List[List[float]]:
        """Compute the (x, y) coordinates of the function over the specified domain.

        Returns:
            A list of coordinate pairs.
        """
        x = self.domain[0]
        # create a `Calculate` object to compute the value of y
        calc = Calculate(0, vars={'x': x})

        # compute y for each value of x in the domain
        while x <= self.domain[1]:
            calc.vars = {'x': x}  # update the calculator variables
            # compute the value of y with the current value of x
            try:
                y = calc.computeExpression(self.function)
            except ZeroDivisionError:
                # asyntope here
                print('zero')
            # if abs(x) < 1:
            #     print(f'x = {x}, y= {y}')

            self.coords.append([x, y])  # append the computed coordinate
            x += self.step
            x = round(x, self.dp)
        return self

    def createPlot(self, otherCoords: Optional[List[List[float]]] = None):
        """Create a plot of the function using the computed coordinates.

        Args:
            otherCoords (List[List[float]]): Optional additional coordinates to plot on the same graph.

        Returns:
            None
        """
        if otherCoords == None:
            otherCoords = []

        plt.plot([0, 0], [self.domain[0], self.domain[-1]])
        plt.axvline(x=0)
        plt.axhline(y=0)
        # plt.set_aspect('equal', adjustable='box')
        # plt.tick_params(axis='both', labelsize=14)

        # print('xs', [x for x in self.coords[0]])
        plt.plot([x[0] for x in self.coords], [x[1]
                                               for x in self.coords])
        for coordSet in otherCoords:
            # print(coordSet)
            plt.plot([x[0] for x in coordSet], [x[1]
                                                for x in coordSet])
        plt.show()


class ParametricGraphing(CartGraphing):
    def __init__(self, functionOfX, functionOfY):
        super().__init__('')
        # functions use param t
        self.functionOfX = functionOfX
        self.functionOfY = functionOfY

    def plotCoords(self):
        t = self.domain[0]
        calc = Calculate(0, vars={'t': t})
        while t <= self.domain[1]:
            # parametrics use one sync var t, so the x and y need to be calculated with respect to this
            calc.vars = {'t': t}
            y = calc.computeExpression(self.functionOfY)
            x = calc.computeExpression(self.functionOfX)
            # print(x, y)
            self.coords.append([x, y])
            t += self.step
        return self.coords


def solveSystemsOfEqNp(vars, equations, answers):
    # vars, the names of the variables, in the order they are given
    # equations: 2d array, containing the coefficients of the variables
    # answers: the value of each equation in the system
    m = np.asarray(equations)
    print(m)
    v = np.asarray(answers)
    v.reshape(len(answers), 1)
    # first check if m is a singular matrix, if it is it has no inverse therefore it has no solutions
    detM = np.linalg.det(m)
    if detM == 0:
        print('no solutions')
        # no solutions
        pass

    # step 1, inverse the matrix
    mInv = np.linalg.inv(m)
    coefs = mInv.dot(v)
    result = {}
    for var, coef in zip(vars, coefs):
        result[var] = coef
    return result


'''In this code, the Matrix class and its Matrix2x2 subclass are defined to represent and manipulate matrices. The __init__ function initializes an instance of the Matrix class and creates an empty list to store the elements of the matrix. The addEquation function adds a row of coefficients to the matrix, and the dot function performs a dot product of two matrices. The scalar function multiplies all elements of the matrix by a scalar value, and the hasValidSolutions function checks if the matrix has a valid solution by checking if its determinant is zero.

The Matrix2x2 subclass extends the Matrix class to support operations on 2x2 matrices, including the getDeterminant function, which computes the determinant of a 2x2 matrix, and the invert function, which inverts the matrix.

The data list is used to store the elements of the matrix, with each element of the list representing a row of the matrix. This data structure is used to store the elements of the matrix and support operations on the matrix, such as dot products and inversion. The eq variable is used to store a row of coefficients when adding an equation to the matrix, and the scalar variable is used to store the scalar value that will be used to multiply the elements of the matrix.

These data structures were chosen to support the operations performed by the Matrix and Matrix2x2 classes, including the storage and manipulation of matrices. The data list is used to store the elements of the matrix, and the eq and scalar variables are used to support operations on the matrix.'''


class Matrix:
    def __init__(self):
        self.data = []

    def addEquation(self, eq):
        # eq are of coefficient, as array
        self.data.append(eq)

    def dot(self, mat2: "Matrix") -> List[List[float]]:
        """Compute the dot product of this matrix with another matrix.

        Args:
            mat2 (Matrix): The second matrix to compute the dot product with.

        Returns:
            The resulting matrix from the dot product.
        """
        result = []
        for i in range(len(self.data)):

            result.append([])  # initialize a new row in the result matrix
            row = self.data[i]
            for j in range(len(mat2.data[0])):
                # get the jth column of the second matrix
                col = [mat[j] for mat in mat2.data]
                total = 0
                # compute the dot product of the ith row and jth column
                for v1, v2 in zip(row, col):
                    total += v1 * v2
                result[i].append(total)
        return result

    def scalar(self, scalar: float):
        """Multiply all values in the matrix by a scalar value.

        Args:
            scalar (float): The scalar value to multiply the matrix by.

        Returns:
            Non"""
        for x in range(len(self.data)):
            for y in range(len(self.data[0])):
                # multiplies all values in the matrix by a number
                self.data[x][y] = self.data[x][y]*scalar
                # self.data[x][y] *= scalar

    def hasValidSolutions(self):
        # if det of a matrix = 0 the system has no solution
        if self.getDeterminant() == 0:
            return False
        return True


class Matrix2x2(Matrix):
    def __init__(self):
        super().__init__()

    def getDeterminant(self):
        # minor: a small 2x2 matrix
        a = self.data[0][0]
        b = self.data[0][-1]
        c = self.data[1][0]
        d = self.data[1][-1]
        # each element of a 2x2 matrix
        print(f'a:{a} b:{b} c:{c} d:{d} det: {a*d-b*c}')

        return a*d - b*c  # formula of a determinant

    def invert(self):
        """Invert the matrix by swapping certain elements and multiplying by the reciprocal of the determinant."""
        # Calculate the determinant
        det = self.getDeterminant()
        # Swap the elements of the matrix so they can be inverted
        a = self.data[0][0]
        b = -1*self.data[0][-1]
        c = -1*self.data[1][0]
        d = self.data[1][-1]
        # Replace the elements of the matrix with the swapped elements
        self.data[0][0] = d
        self.data[-1][-1] = a
        self.data[0][-1] = b
        self.data[1][0] = c
        # Multiply the matrix by the reciprocal of the determinant
        self.scalar(1/det)


'''This code defines a class called Matrix2x2 that appears to be a subclass of Matrix. The Matrix2x2 class has three methods: __init__(), getDeterminant(), and invert().

The __init__() method simply calls the parent class's __init__() method, which seems to be used to initialize the data of a Matrix object.

The getDeterminant() method calculates and returns the determinant of a 2x2 matrix using the formula a*d - b*c, where a, b, c, and d are the elements of the matrix.

The invert() method inverts the matrix by swapping certain elements of the matrix and multiplying the resulting matrix by the reciprocal of the determinant. This method assumes that the determinant of the matrix is nonzero, so it will not work on matrices with a determinant of 0.'''


class Matrix3x3(Matrix):
    def __init__(self):
        """Initialize the matrix data by calling the parent class's __init__() method."""
        super().__init__()

    def getDeterminant(self) -> float:
        """Calculate and return the determinant of the 3x3 matrix.

        Returns:
            float: The determinant of the matrix.
        """
        m33 = Matrix3x3()
        for x in range(3):
            currentRow = []
            for y in range(3):
                # Indices of the elements to include in the minor matrix
                if x == 0:
                    x1 = 1
                    x2 = 2
                if x == 1:
                    x1 = 0
                    x2 = 2
                if x == 2:
                    x1 = 0
                    x2 = 1
                if y == 0:
                    y1 = 1
                    y2 = 2
                if y == 1:
                    y1 = 0
                    y2 = 2
                if y == 2:
                    y1 = 0
                    y2 = 1
                # Create the minor matrix and calculate its determinant
                m = Matrix2x2()
                m.addEquation([self.data[x1][y1], self.data[x2][y1]])
                m.addEquation([self.data[x1][y2], self.data[x2][y2]])
                minor_det = m.getDeterminant()
                # Add the determinant to the current row, negating it if the sum of x and y is odd
                if (x+y) % 2 == 1:
                    currentRow.append(-1*minor_det)
                else:
                    currentRow.append(minor_det)
            # Add the current row to the matrix of minors
            m33.addEquation(currentRow)
            currentRow = []

        # The determinant of a 3x3 matrix is the sum of the elements of the matrix of minors,
        # with the elements in the first row multiplied by the corresponding elements in the original matrix
        a = m33.data[0][0] * self.data[0][0]
        b = m33.data[0][1] * self.data[0][1]
        c = m33.data[0][2] * self.data[0][2]
        return a-b+c  # det of the first minor - det second minor + det third minor

    def invert(self) -> "Matrix3x3":
        """Invert the 3x3 matrix using the determinant and the matrix of minors.

        Returns:
            Matrix3x3: The inverted matrix.
        """
        # Calculate the determinant of the matrix
        det = self.getDeterminant()

        # Create the matrix of minors
        m33 = Matrix3x3()
        for x in range(3):
            currentRow = []
            for y in range(3):
                # Indices of the elements to include in the minor matrix
                if x == 0:
                    x1 = 1
                    x2 = 2
                if x == 1:
                    x1 = 0
                    x2 = 2
                if x == 2:
                    x1 = 0
                    x2 = 1
                if y == 0:
                    y1 = 1
                    y2 = 2
                if y == 1:
                    y1 = 0
                    y2 = 2
                if y == 2:
                    y1 = 0
                    y2 = 1
                m = Matrix2x2()
                m.addEquation([self.data[x1][y1], self.data[x2][y1]])
                m.addEquation([self.data[x1][y2], self.data[x2][y2]])

                # step 3
                # +-+ matrix
                if (x+y) % 2 == 1:
                    # all even sums of x,y make their element negative
                    currentRow.append(-1*m.getDeterminant())
                else:
                    # otherwise it is just a positive element
                    currentRow.append(m.getDeterminant())

            m33.addEquation(currentRow)
            currentRow = []
        self.data = m33.data
        print('INVERT MATRIX')
        print(self.data)

        # step 4
        # transpose the matrix
        transposedMatrix = [[], [], []]
        for x in range(3):
            row = self.data[x]
            transposedMatrix[0].append(row[0])
            transposedMatrix[1].append(row[1])
            transposedMatrix[2].append(row[2])

        self.data = transposedMatrix
        # step 5:
        # finally 1/det * transposed
        self.scalar(1/det)
        res = Matrix3x3()
        for i in range(3):
            res.addEquation(transposedMatrix[i])
        return res


'''This code defines a class called Matrix3x3, which is a subclass of Matrix. The Matrix3x3 class has two methods: __init__() and invert().

The __init__() method is similar to the one in the Matrix2x2 class, in that it simply calls the parent class's __init__() method to initialize the matrix data.

The invert() method is used to invert a 3x3 matrix. It follows these steps:

Calculate the determinant of the matrix using a method called getDeterminant(), which is not defined in this code snippet.
Create a 3x3 matrix called m33 containing the minor matrices of the original matrix. The minor matrix of an element is the 2x2 matrix formed by the elements in the same row and column as the element, with the element itself removed.
Invert m33 by taking the determinant of each of its minor matrices, multiplying each element by -1 if the sum of its row and column indices is odd, and replacing each element of m33 with its inverted value.
Transpose m33 by swapping the rows and columns.
Multiply the transposed matrix by the reciprocal of the determinant of the original matrix.'''


def solveSystemsOf2Eq(equations, answers):
    m1 = Matrix2x2()
    for equation in equations:
        m1.addEquation(equation)
    print('MATRIX DATA')
    print(m1.data)
    m2 = Matrix2x2()
    # m2.addEquation(answers)
    for a in answers:
        m2.addEquation([a])
    if m1.getDeterminant() == 0:
        return None
    m1.invert()
    print('matrix 1 ', m1.data)
    print('matrix 2 ', m2.data)
    return m1.dot(m2)
    # vars, the names of the variables, in the order they are given
    # equations: 2d array, containing the coefficients of the variables
    # answers: the value of each equation in the system


def solveSystemOf3Eq(equations, answers):
    m1 = Matrix3x3()
    for equation in equations:
        m1.addEquation(equation)
    m2 = Matrix3x3()
    for a in answers:
        m2.addEquation([a])
    if m1.getDeterminant() == 0:
        return None
    m1.invert()
    return m1.dot(m2)


'''In this code, the Matrix class and its Matrix2x2 subclass are defined to represent and manipulate matrices. The __init__ function initializes an instance of the Matrix class and creates an empty list to store the elements of the matrix. The addEquation function adds a row of coefficients to the matrix, and the dot function performs a dot product of two matrices. The scalar function multiplies all elements of the matrix by a scalar value, and the hasValidSolutions function checks if the matrix has a valid solution by checking if its determinant is zero.

The Matrix2x2 subclass extends the Matrix class to support operations on 2x2 matrices, including the getDeterminant function, which computes the determinant of a 2x2 matrix, and the invert function, which inverts the matrix.

The data list is used to store the elements of the matrix, with each element of the list representing a row of the matrix. This data structure is used to store the elements of the matrix and support operations on the matrix, such as dot products and inversion. The eq variable is used to store a row of coefficients when adding an equation to the matrix, and the scalar variable is used to store the scalar value that will be used to multiply the elements of the matrix.

These data structures were chosen to support the operations performed by the Matrix and Matrix2x2 classes, including the storage and manipulation of matrices. The data list is used to store the elements of the matrix, and the eq and scalar variables are used to support operations on the matrix.'''


class BinomialDist:
    def __init__(self, n, p):
        """Initialize the binomial distribution with the given number of trials and probability of success.

        Args:
            n (int): The number of trials.
            p (float): The probability of success on a single trial.
        """
        self.n = n
        self.p = p

    def single_value_probability(self, x: int) -> float:
        """Calculate the probability of a single value in the binomial distribution.

        Args:
            x (int): The value for which to calculate the probability.

        Returns:
            float: The probability of the given value.
        """
        # calculate the probability of a single value
        return self.combination(self.n, x) * (self.p ** x) * ((1 - self.p) ** (self.n - x))

    def cumulative_probability(self, x1: int, x2: int) -> float:
        """Calculate the cumulative probability of a range of values in the binomial distribution.

        Args:
            x1 (int): The lower bound of the range of values.
            x2 (int): The upper bound of the range of values.

        Returns:
            float: The cumulative probability of the given range of values.
        """
        # make sure x1 <= x2
        if x1 > x2:
            x1, x2 = x2, x1

        # calculate the cumulative probability
        prob = 0
        for i in range(x1, x2+1):
            prob += self.combination(self.n, i) * \
                (self.p ** i) * ((1 - self.p) ** (self.n - i))

        return prob

    def factorial(self, n: int) -> int:
        """Calculate the factorial of a given number.

        Args:
            n (int): The number for which to calculate the factorial.

        Returns:
            int: The factorial of the given number.
        """
        if n == 0:
            # The factorial of 0 is 1
            return 1
        else:
            # Compute the factorial by multiplying n by the factorial of n - 1
            return n * self.factorial(n - 1)

    def combination(self, n: int, k: int) -> float:
        """Calculate the number of ways to choose k items from a set of n items.

        Args:
            n (int): The total number of items in the set.
            k (int): The number of items to be chosen.

        Returns:
            float: The number of ways to choose k items from a set of n items.
        """
        # calculate the combination (n choose k)
        return self.factorial(n) / (self.factorial(k) * self.factorial(n - k))


if __name__ == '__main__':
    # param = ParametricGraphing(
    #     '2*cos(t) + 5*cos((2/3)*t)', '2*sin(t)+5*sin((2/3)*t)')
    # param.plotCoords()
    # param.createPlot()
    # trueSin = [[x/10, 10*math.sin(x)] for x in range(-100, 100)]
    # c = Calculate({}, {'x': 1})
    # print(c.computeExpression('sin(x)'))
    g = CartGraphing('10*sin(x)')
    coords = g.plotCoords()
    # # print(coords)
    g.createPlot()

    b = BinomialDist(50, 0.5)
    print(b.cumulative_probability(10, 30))
# class CreateCartGraph


['-10+9--5+-6', '1-2+2/3/8', '5+-3/-7', '5+-5/-3-3-7', '-10/-9-0-5-6', '7+7+8-1/-7', '-9-4-7', '10/-5+4-7*-3/-8', '8*-3-2', '8-0+1/5+-10', '2*0+-2*1+3', '-4/-4+-5*-5*0', '-10-9+4+8', '-9--6*-6+-5+-7-9', '-4--3--8-5', '-6*5/-4+-6+-3', '-7*6+-10/-2+-6', '-6/-8-4/-8', '-7+0-8-4', '9*-5-5/1', '-6--9-1*-6', '-4/-5+2/-8+-4--9', '5/9+-9*-6+-2*5', '-3-9+-6+0', '3+-7--9/9/-3', '-4+-7+3', '3/-2*-5+-1+-6', '2/1*0-7--10/3', '-9+-6--3', '-6-5*3-9', '10*-4+-8', '-5+7-10', '2+-7--4*4', '-1--7/-3+-5', '-10*-8+-5/-10--5', '9*-9-5--9/-2--9', '-3*-10/-8+-5+-8', '7*5+8*-10+-1', '6+-10/8*-2', '7+-6/1', '5*-3/10*10+6+-2', '-3*-3+-8', '-10+-8+-3+-6+-3', '4+8/-3-7-9', '-2+2*3-7', '4-10+-6+-7--1', '4/2-4', '-10/-10+-8*1*-7', '3--9+-6*9*7', '7+0+-8*3', '6/-9*-9--2-6', '-7/9*6*-1/-5-9', '10/6-8+-6/4', '-5-9+1*4*6', '0*6-3/7', '2*1+-9--9', '4-8*2-3/-9--5', '-9+0/4/3+-7', '4-3--4+10', '-1+0+-2*9/10', '-8--6+9*6+-3', '7+-9-1+0*7', '-9*1-7', '2/10/-5/7+-3', '4--10+-5', '-5*0+-7--5+-5+-6', '3+-5-8*7+3', '-7+-3/8/4+1*-9', '-4*-9+-7/-6-3', '-7-9*8*-3', '-1-10/6*4+-10', '-2/-2/4-6--3', '4-6*-4--5', '9+-4*-1+-4*7', '-10*10+-7*5*8-8', '10+-3/5-10--4*-5', '8+-3/3/-10--7', '8--2*0/1+-2+0', '2*1*-1-10/-8', '-8+-8*7+-3-9', '0--9/8+-2/8', '-2*-9/8+-5', '5+-10+-6--9*2', '9--2*9*10+-8/-8', '7/-8+1/-9-3', '9*-7+4*8-2', '4+-9--9/3+-10/8', '3+-10*6+3+8', '5+-3+-6+7+6', '-1/-9/2+-9+4*-10', '-3+7/6-4', '-8/10-5', '-1+-5/-5+1--7-2', '2/3-4*-6', '-2+-7-8*9+8', '3+-2*-2*-8', '1-4-1', '9+5--7+-8+-10', '10+-4/-8/4', '3*-1/-8*-5+10-2', '0+-10*2+0-10/9', '8*2+-9', '1*-10+-2-0', '5-7--6*-1', '-2+3+-7', '0*-6+9-7+-4/-6', '-1*-9+-8-0/6', '-7*-6/5/-7-4', '-9+-7*1+-3', '8-10+4-2+-3', '10/-3/-6+3-1', '-4*3*-6-2+-6', '6/3+-2*-1*3', '7+5-5/1-6', '-5*3--5+-1/9--3', '2/5+-7*-3*-5', '-8/-3/-6/-4+-2+-6', '-2/7--5+-5*7', '-3/-5+-10/-8', '7-9*-9/6', '4--7+-6', '-8-8/-3', '4*2+-7*10--9*6', '-1+-6+-7-10*9*3', '-2*0-6*8*8+-4', '-1/-9+-10+-5+2', '3+6+-8*-7*6', '-1-5+-4*-7*-8', '4*6+-6*9', '2-9*4+-2', '10/5--3+-8/1', '9--2-7', '-2/4-2+-2', '5*-2-7', '7+-7*2*-4', '4-9--6/4-5', '6+0-7*1', '0-10+9*9/8-7', '-3--9-6--10+2', '-7+-6*1', '10*-6-1--8', '-8/5*-8-2--10', '1+-5+8*-3', '-10+10+-1/1', '4*-10+-10', '-10-3/5/-2*6', '-2--10--9+4+-5', '-4*-1+-9--8/-9', '-5/-9-4/-3*8*5', '-3-5-0*-8-8', '-9--1-3/-6', '-9-4-6*-10', '4*7+-5*-10', '0--5/-7*6-1', '6+2--2*-7-0--3', '-4*-3+9/-9-7/-7', '5+3+-6', '8+4-7/-2--2*5', '2/-4-10/-9/4', '-2-2/4+-9', '-6-5-5*-4*10*4', '-9-9--4+-1/10', '-1/-6-3+2+-4+-2', '-8--8/-9+-4', '-6--2/-2-7*-3/4', '-9/4/-9/4+-6', '-2*5*7-3*3', '-1+-10/-9--4/-9*9', '10-4/-4/3--6', '9+-3/10', '9-1+-6--5+-10*-9', '8*1-3/3/-8*0', '-8/10+-7/6-5', '7-4+6*0/8/7', '6-3*7*-6', '-2/-9+5--8--5-5', '1/-7--2*-8+-4', '10-1*8--9', '-9/-8*-4+-7', '6/1/10/9/4-8', '10/9--7+-8/4', '-5-5*-2', '-5/-10--2+-7-6/-9', '9/-5*-8-10', '-9/6-1--8-5*5', '7/10-2', '6--2+-10/-5*10', '-10--4*-4*0-3', '7*-6+-10/7', '-2--3*9-8--7', '-7--6/10-8/10', '1+-7--8', '4*3*5+-4', '-10/-4/1-6', '-4*7--2+-6', '-3*-4*9*5-2', '-5--8-7', '10/-7*10-10', '4/2+-1', '-6-4/-5*-5', '-6+2--4-3/-5', '9-10*3*-3/-6/9', '2*4*8*-7-2', '-10+-6*4', '9/-3+-10+0-7/-5', '7-3+1+-5+3', '4-4*0+7', '1/5--2*8-7', '-1/-4+-4/6--8', '-1+5-8*5/2', '4--2+-9/-2', '0-5-8+-6-8', '7-2*5', '4*4-8', '-9+-7-8-9*7', '10-8+8*-10', '3*4*5-9--4+9', '3--1-2/-7/4/-3', '-4*-7+-3-10', '-5/3--7/6*-10+-6', '4*-4+-9/-9+3', '-4+-8*-4', '10/-2*-5-9-1/-10', '9/5+-8/-7+9', '6*7*3-3+-5+-9', '9-6*9', '1+-5/-2', '6+0-7*-7*-7', '3+-1*-3*-10--2', '1-9-7/-5+3+-3', '-3-0*9+-6', '9*-3+8-8+-10', '-4-9-6+9/-8', '8--9-2+10-10', '4+-5/10-7', '-7/-4*-5+-9', '4/-4/3-8', '1/5/10+2--6-0', '-8-10*9*-3/6', '1+-9/5*6--4', '5-10+-10/-5', '1/9+-7/-9*3', '-7+-9*-2--1', '3+-10/10-4-7', '10-4/-4-6*6', '-7*6-3-3', '-10+-5+-3*-9', '3-2+10-7', '7-3*-6*-8', '8+-10--1+5', '5--9-8-5-2', '-8+-5*0-9--7--4', '-7/-6+-4', '1*6+-6*-6', '-5/5/-2-10/-3--8', '7+6+-3*-3+6', '10--1-0+4', '9/-5+-4-10*9/-1', '8/4+-3*2', '-2--8+-4+-3', '1/2+-3', '-9/3-5+1+-4', '2*-8-9', '7/10+-9', '8-10/-6/-1--1/10', '-5/5-1+10', '10+-5-0/10/-2', '2/5+-10-3/-4+2', '-8-0+-7*-9+8--4', '-10/9+4+-7+5--8', '-2-2*-2-3-2', '-10*-7*4*-6+-5', '3--8*-1*10-1', '-2/8+3-5*-7', '-7/9*10-3', '7*3/2+-7/-3+10', '-10+3-10-8-7+4', '7--8-2*4', '0/-6*-6+-4*9', '1--1*9*5-6', '-10/-1-6*6-10--7', '10+2+-3-5/-10/6', '2*-7--1+-5', '1-10/-3/-7-9', '4*10/3*-7+-8', '-1/5-10/8/7', '-5+-9+-1/8', '0/-5*5+-7*-10', '9*0+-4--10', '1*4+-2--9', '-6-3+3', '6-5/-6*3-9', '-5+-6*6/-9-9', '9+-1--1--10*1', '5-8-1-2+-3', '6/9*5/-10-9', '1-4*-2/-3/-3*10', '-5/6-0+-8--9', '-6/2/7-8', '4*4+-6', '8--10*-8+-9--9', '4+-5-1*4', '-6+1+-3', '2/-9*-5-5', '7/4-1*3*-7', '4/3+-5', '7*3*8-7', '8+8-2/-1+6+-9', '-3-2--9', '-2/9-10*-4', '1+-4+-1*1', '5+-8/9+-4--2', '-2/-6*-4-0', '4/-10-8+-9--2', '-10/9+-3/1', '-10-2+4-5/-7', '-6-0/-10/6*3', '4/-9--1-7*-4', '6*5-10', '-1+-7/-10-5/-5', '-3--9*9*7+-1', '9+-6/9/-7', '-1+8-2*-7*3', '-8*1-8/7+2/-5', '-2-2/-7', '-3*2-8+-8', '-4+5-2*4--10+8', '0*-3+2+4-10', '4+4--6-7*-4*-5', '-10*3-2+1/-10*-7', '-2*-10+-3', '-3-5*0+-5+3', '7+6-2+-8', '-9*8/4-10+-1', '0-1/-7--1--2-10', '5*-2+-2*-7/-7', '-10*-4/5+-7/-1', '10--1+-6/7', '9/8+-2*-6/2', '6*5*-5-3*4-7', '-9+1+-2-3', '3/-5-7/-10-4-1', '-4+-10--8', '-1--4/-10+-5', '10+-1--8/-8/-4', '5+5/-6-1', '2/5--1+-7*4', '10/9-8-5*1', '3-4+4', '-10--8+-6', '7+-6+-4-7*-2', '4+-6-8', '-3-2/9+-4', '7+-4*9*-3/1', '-9*3+8+-8*5-9', '-1+-6--9', '-7*-3-9*9*5/-8', '-3--4--5-2+-2*-9', '-6--7--7-0-10', '-6*6+-7/4', '8*-3-6-0', '3-3/-10-2', '8-4/-7+9', '-2+0*-4+-7+5', '7/-10-3*-7--5--2', '-4+3--4-0', '6*-2--5+-10', '-1*-3+-4*10--8', '-9--1/2/-4+-10', '8-1+-7', '-10*-6+-6--5/-10', '-4-9*-2*1--4--5', '-7+-2/3--1', '-6-0+-4--2/-6', '-4+-5+-7-2--4*-5', '-8/8+-7*-7*-8', '-7-1*5--1+2-7', '3+3-3-7', '-6-1+0*5*4', '8--1-7+5+-7', '-1--8+-3/1', '8/-8+-10', '10*-10+-1/-9*10', '10+-3/5*-3*2', '-4--9+-8', '-3-9/6', '10/-10*-10-8', '2*0--7--1+-7/-4', '1--7-0/-4-9+-6', '-8+-9*8--9', '4*-9+-1+3', '4-4*1-10-4+7', '10*-7+-9--10*2', '6-0+-9+10/5', '1+-9+5/5', '6+-10*-8--7', '-9+-8-10--5', '-5*1-4/9*0', '4/-9-5*10+3', '6+-4+-1-2/-6', '7-2*-1+2', '4--8+-7--2*-9', '0+5/-3-3/6*6', '2-2/-5*-6+3', '-5*9+-4+6*-10*-5', '-10-0+9', '5+-1-2*-7*6', '4+5+-2', '3*-1+-3--3', '-8+1+10/-2+-6+-5', '-2+-5*7/-8', '3+8+0+-9+-7/5', '7+-7*-6+-8', '-4+-8/10*-3', '-6/1+-3-1', '-2+-3--3', '-7*-5*-4+-8-8+3', '10*4*-8+-8/10*-7', '9*3*8--1+-2*-4', '-8/2-5+10+-2', '10+0-0+8--2', '-7-6--9--10', '-9/5+-10--7', '9+-7*5*7', '6--1*0+-7', '-2*4-8-8', '5*-5+-4', '-4/-7+-5-6', '2+-1+-8+-9/4', '9/2-10--4', '5*-5-10/8/10*3', '-4*-2/10+1+-7*5', '7+-4*-8/-2', '-10+-2*2--5', '10/2/-8-10', '4-3/6-3', '-1+-10*-1', '-1--3-2/-7', '-3+2--4/7+6+-4', '-9*5-10*0-10/4', '10+-1/10+9', '5+2*0+7+-5', '-7/8*-8/-1-10', '-3-4*-4--9/-5', '-2+-7*8', '3+-9/-1-5', '-7/-8-7+10', '-9--10-6-7-6-0', '-1/5--8-9', '5-9-4/-4', '1+10-1-5', '-7--1+1-9/9', '8+-6*-9/3+-5', '6/8+-1', '-6+-6/6/1', '5/6/1*-3/-6-10', '2/-1*6-5--10', '8*10-10--3', '-6-1-9/8+-5', '-9/8/1/-6+-2/7', '1+1+-8', '-9/-7*6-9-1', '9+-7-8/10-6*5', '3-9+7*9', '-7/2-5+-4', '6-2+-6*3', '7/-6/-8--1--7+-10', '10-4/-4/-4+-2', '2+-4/3*-10', '6*5-2-7+8', '-8-9--1+2/-4', '-7+-9*-6*-9', '-2-6+-8+3', '5*-1/4-4', '8-2/1+1', '-6*-4+-9--10', '-8-10*1+-6', '-4-10-7/5', '1--2+9-10', '-3/-3+-4/9+6', '7+5/6-8+-3', '-7*8+7--10-9/3', '3+7*9-8--8', '9/-1+5-6', '5-4*9', '-4+3+4*0/5+-5', '-3+-10+2-1', '-6+7-6-7', '-5-9--8*-7--8*-8', '2/6+-1', '-9--4-1*-3/-1', '-4+-3+7/-9/-9-7', '8/2-9/4*3', '4-8+-9*-4+10+-2', '6+-6/4', '4+-3/4', '-3-5-6-9', '-4+10/10+-6/5/2', '8*2-0+-8', '1--4+-6--6*-9', '-8/-10+-8+-1', '6+-5/6--8-7/-4', '4+7*9-6', '9+-1*-1--4*4', '4*-5*-6+-3', '5/-3+-8+-6*-10-7', '8+-4+-3',
 '-1+4+3+-4*10--8', '6/-2+-5/7', '4*5+-3+6', '10-0*0--4*3--7', '6/3/-3+-4', '10-8*-7+4+-5*-10', '2-9-8+-10*-3+-9', '10*-6+3+-4+10', '9+1*2--2-8--5', '-5+-2+6-3/-10', '2-1/-6/9', '2+-5*10*1', '5*-9+-10+10', '0+1/2+-1*-1-1', '4*1/-3-9+7', '0/-2/4*8+-3', '8-5/-2', '2+9*-1/8-3', '2*10+-3+-8', '8+7/-7*-3+-9', '8--4*-5+5+-10+-3', '8+0+-9/10*-6', '-5*-6/3*4/-8-3', '-8--2*3-6/-10+-7', '1+-7*-9--7+-8', '-1+-4-8--1+-3', '8/5*1-1', '10/-6+-3', '-6*5-2/5', '6/-9/5*-3+-8', '-2/10+-7*5+6', '8+-5/-10--4', '-5+5/4+-1+-2', '-9*9/-6+-10', '-7/10+-9*10', '-4/3--1+6+-10', '1+-5*3', '-6+-1/4-6', '4-8-5*-6--10', '-2-10/9', '-5+-5-0--2', '-1+0+-3*-5/-8', '4+-9/3--5', '-10/-9*-9-10*5', '7*-6*4/2+-1', '1+0+-5+6', '-7/4+-10*-5--8', '-6--4/-4-3--9',
 '5+-2*6+7', '-1+3+-5*1', '6/-3*-9--7*-9+-6', '3+-3--3+8', '0+7/-5+-4-3', '8/-8+-4/7', '10*5+-2*1*-7', '-2+-2--8+8', '-7+-8--10*7*-1+2', '-10/4-3', '-4/3/-1+-9', '5/10-6+5/4', '-1-9*-7', '4+-1+-9*-10', '0/10+-4+6', '-1+8*2-0*9', '6-5+6--1-8', '-6*-5-2-0', '8--4--8+-6', '7+2-2--5/-5', '10--3*-10+-9/-10', '10/3-4+1+-8', '-4*-3+-9*9', '2/-6*-5-5/-10--9', '-4+-9+-10+-5', '-7+-1-10', '9/2-6', '4-7/1/8', '8*6-7-4--7/5', '7/-9+1+-1*8--9', '-4+-8--2+3/-8', '-10*0-10--8/5', '-1--8--1/9-4-10',
 '9+-2/-6+8+-8', '4+1+-2/10*9/5', '-10-8-9*9+6--5', '-5--8/-5+-5-1', '-1+-2+4/2', '1+5*8-7+0', '7-9-5', '2*-8/1+-5*-5', '-8*-9/-3/-1--3+-7', '-3*-3*4+-8+-9', '-9+-7/-5', '8-0+-10/-7*4+8', '2+-3*-5--1', '0+-8+-8*0', '5/-8+-9+10--10', '-3/-5-9-10', '8--10--9+-2', '7/10/-9+-9+8/7', '6/-6--10-8+-1*-1', '6/4-2']
