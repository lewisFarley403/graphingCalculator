from rpn import RPN
from config import precedence
from utils import Stack
import utils
import math
import numpy as np
import matplotlib.pyplot as plt
import math


class Calculate:

    def __init__(self, config: dict, vars=None):
        # vars = None because mutable kwargs in pyton are created when the class is first interpretted so this mutable dictionary is shared for all instances, that is why it is none and a dict is created if one isnt supplied

        if vars == None:
            self.vars = {}  # letters that have a value attatched to them
        else:
            self.vars = vars
        self.vars['π'] = math.pi

    def computeExpression(self, expression):
        r = RPN(expression, self.vars)  # convert infix to rpn

        return self.computeRPN(r)  # computes the value of the expression

    def computeRPN(self, RPN):
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

    def findInput(self, op1):
        # gets value out of saved vars if it is not a number
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

    def computeOp(self, op1, op2, symb):
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

    def computeTrig(self, func, x):
        x = self.findInput(x)
        if func == 'sin':
            # return math.sin(float(x))
            return self.sin(float(x))
        elif func == 'cos':
            # return math.cos(float(x))
            return self.cos(float(x))
        else:
            return math.tan(float(x))

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

    def sin(self, x):
        # return self.trigBuilder(x, lambda i: 2*i+1)

        # print(f'in sin function: x={x} sin(90) = {math.sin(math.pi)}')
        return round(math.sin(x),4)

    def cos(self, x):
        # return self.trigBuilder(x, lambda i: 2*i)
        return round(math.cos(x),4)


class CartGraphing:
    def __init__(self, function):
        self.function = function
        self.coords = []  # xy coords
        # self.domain = [-100, 100]
        # self.domain = [-250, 250]  # the values x or t take
        self.domain = [-100, 100]  # the values x or t take
        self.dp = 3
        self.step = 1e-3

    def plotCoords(self):
        x = self.domain[0]
        # computes the value of y, the dictionary allows the value it uses for x to change going across axis
        calc = Calculate(0, vars={'x': x})
        while x <= self.domain[1]:

            calc.vars = {'x': x}  # updates the calculator variables
            # computes the value with that value of x
            try:
                y = calc.computeExpression(self.function)
            except ZeroDivisionError:
                # asyntope here
                print('zero')
            # if abs(x) < 1:
            #     print(f'x = {x}, y= {y}')

            self.coords.append([x, y])  # appends that coord
            x += self.step
            x = round(x, self.dp)
        return self.coords

    def createPlot(self, otherCoords=None):
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


# class Matrix:
#     def __init__(self):
#         self.data = []
#     def addEquation(self,coefficients):
#         self.data.append(coefficients)
#     def determinant3x3(self):
#         pass
#     def determinant2x2(self,minor):
#         #minor: a small 2x2 matrix
#         a=minor[0][0]
#         b=minor[1][-1]
#         c = minor[0][-1]
#         d=minor[1][-1]
#         return a*d -b*c #formula of a determinant

class Matrix:
    def __init__(self):
        self.data = []

    def addEquation(self, eq):
        # eq are of coefficient, as array
        self.data.append(eq)

    def dot(self, mat2):
        # dot product of two matrices
        result = []
        for i in range(len(self.data)):

            result.append([])
            row = self.data[i]
            for j in range(len(mat2.data[0])):
                col = [mat[j] for mat in mat2.data]
                total = 0
                for v1, v2 in zip(row, col):
                    total += v1*v2
                result[i].append(total)
        return result

    def scalar(self, scalar):
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
        det = self.getDeterminant()
        # swapping the elements around so that they can be inversed
        a = self.data[0][0]
        b = -1*self.data[0][-1]
        c = -1*self.data[1][0]
        d = self.data[1][-1]

        self.data[0][0] = d
        self.data[-1][-1] = a
        self.data[0][-1] = b
        self.data[1][0] = c

        self.scalar(1/det)


class Matrix3x3(Matrix):
    def __init__(self):
        super().__init__()

    def getDeterminant(self):
        # determinent of a 3x3 mat requires the matrix of minors, creating the matrix of the 2x2 matrices that make it up
        element = self.data[0][0]

        # first minor
        minor = Matrix2x2()
        minor.addEquation([self.data[1][1], self.data[1][-1]])
        minor.addEquation([self.data[2][1], self.data[2][-1]])
        a = minor.getDeterminant()*element

        # second minor
        element = self.data[0][1]
        minor = Matrix2x2()
        minor.addEquation([self.data[1][0], self.data[1][-1]])
        minor.addEquation([self.data[2][0], self.data[2][-1]])
        b = minor.getDeterminant()*element

        # third minor
        element = self.data[0][2]
        minor = Matrix2x2()
        minor.addEquation([self.data[1][0], self.data[1][1]])
        minor.addEquation([self.data[2][0], self.data[2][1]])
        c = minor.getDeterminant()*element

        return a-b+c  # det of the first minor - det second minor + det third minor

    def invert(self):
        # step 1:
        # find deteriment
        det = self.getDeterminant()

        # step 2
        # matrix of minors
        m33 = Matrix3x3()
        for x in range(3):
            currentRow = []
            for y in range(3):
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
        # print('TRANSPOSED ENGINE')
        # print(transposedMatrix)
        # print('1/det ',1/det)
        # print(f'LENS {len(self.data)} {len(self.data[1])}')
        self.data = transposedMatrix
        # step 5:
        # finally 1/det * transposed
        self.scalar(1/det)
        res = Matrix3x3()
        for i in range(3):
            res.addEquation(transposedMatrix[i])
        return res


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
    m1.invert()
    return m1.dot(m2)


if __name__ == '__main__':
    # param = ParametricGraphing(
    #     '2*cos(t) + 5*cos((2/3)*t)', '2*sin(t)+5*sin((2/3)*t)')
    # param.plotCoords()
    # param.createPlot()
    trueSin = [[x/10, 10*math.sin(x)] for x in range(-100, 100)]
    c = Calculate({}, {'x': 1})
    print(c.computeExpression('sin(x)'))
    g = CartGraphing('10*sin(x)')
    coords = g.plotCoords()
    # print(coords)
    g.createPlot([trueSin])


# class CreateCartGraph


['-10+9--5+-6', '1-2+2/3/8', '5+-3/-7', '5+-5/-3-3-7', '-10/-9-0-5-6', '7+7+8-1/-7', '-9-4-7', '10/-5+4-7*-3/-8', '8*-3-2', '8-0+1/5+-10', '2*0+-2*1+3', '-4/-4+-5*-5*0', '-10-9+4+8', '-9--6*-6+-5+-7-9', '-4--3--8-5', '-6*5/-4+-6+-3', '-7*6+-10/-2+-6', '-6/-8-4/-8', '-7+0-8-4', '9*-5-5/1', '-6--9-1*-6', '-4/-5+2/-8+-4--9', '5/9+-9*-6+-2*5', '-3-9+-6+0', '3+-7--9/9/-3', '-4+-7+3', '3/-2*-5+-1+-6', '2/1*0-7--10/3', '-9+-6--3', '-6-5*3-9', '10*-4+-8', '-5+7-10', '2+-7--4*4', '-1--7/-3+-5', '-10*-8+-5/-10--5', '9*-9-5--9/-2--9', '-3*-10/-8+-5+-8', '7*5+8*-10+-1', '6+-10/8*-2', '7+-6/1', '5*-3/10*10+6+-2', '-3*-3+-8', '-10+-8+-3+-6+-3', '4+8/-3-7-9', '-2+2*3-7', '4-10+-6+-7--1', '4/2-4', '-10/-10+-8*1*-7', '3--9+-6*9*7', '7+0+-8*3', '6/-9*-9--2-6', '-7/9*6*-1/-5-9', '10/6-8+-6/4', '-5-9+1*4*6', '0*6-3/7', '2*1+-9--9', '4-8*2-3/-9--5', '-9+0/4/3+-7', '4-3--4+10', '-1+0+-2*9/10', '-8--6+9*6+-3', '7+-9-1+0*7', '-9*1-7', '2/10/-5/7+-3', '4--10+-5', '-5*0+-7--5+-5+-6', '3+-5-8*7+3', '-7+-3/8/4+1*-9', '-4*-9+-7/-6-3', '-7-9*8*-3', '-1-10/6*4+-10', '-2/-2/4-6--3', '4-6*-4--5', '9+-4*-1+-4*7', '-10*10+-7*5*8-8', '10+-3/5-10--4*-5', '8+-3/3/-10--7', '8--2*0/1+-2+0', '2*1*-1-10/-8', '-8+-8*7+-3-9', '0--9/8+-2/8', '-2*-9/8+-5', '5+-10+-6--9*2', '9--2*9*10+-8/-8', '7/-8+1/-9-3', '9*-7+4*8-2', '4+-9--9/3+-10/8', '3+-10*6+3+8', '5+-3+-6+7+6', '-1/-9/2+-9+4*-10', '-3+7/6-4', '-8/10-5', '-1+-5/-5+1--7-2', '2/3-4*-6', '-2+-7-8*9+8', '3+-2*-2*-8', '1-4-1', '9+5--7+-8+-10', '10+-4/-8/4', '3*-1/-8*-5+10-2', '0+-10*2+0-10/9', '8*2+-9', '1*-10+-2-0', '5-7--6*-1', '-2+3+-7', '0*-6+9-7+-4/-6', '-1*-9+-8-0/6', '-7*-6/5/-7-4', '-9+-7*1+-3', '8-10+4-2+-3', '10/-3/-6+3-1', '-4*3*-6-2+-6', '6/3+-2*-1*3', '7+5-5/1-6', '-5*3--5+-1/9--3', '2/5+-7*-3*-5', '-8/-3/-6/-4+-2+-6', '-2/7--5+-5*7', '-3/-5+-10/-8', '7-9*-9/6', '4--7+-6', '-8-8/-3', '4*2+-7*10--9*6', '-1+-6+-7-10*9*3', '-2*0-6*8*8+-4', '-1/-9+-10+-5+2', '3+6+-8*-7*6', '-1-5+-4*-7*-8', '4*6+-6*9', '2-9*4+-2', '10/5--3+-8/1', '9--2-7', '-2/4-2+-2', '5*-2-7', '7+-7*2*-4', '4-9--6/4-5', '6+0-7*1', '0-10+9*9/8-7', '-3--9-6--10+2', '-7+-6*1', '10*-6-1--8', '-8/5*-8-2--10', '1+-5+8*-3', '-10+10+-1/1', '4*-10+-10', '-10-3/5/-2*6', '-2--10--9+4+-5', '-4*-1+-9--8/-9', '-5/-9-4/-3*8*5', '-3-5-0*-8-8', '-9--1-3/-6', '-9-4-6*-10', '4*7+-5*-10', '0--5/-7*6-1', '6+2--2*-7-0--3', '-4*-3+9/-9-7/-7', '5+3+-6', '8+4-7/-2--2*5', '2/-4-10/-9/4', '-2-2/4+-9', '-6-5-5*-4*10*4', '-9-9--4+-1/10', '-1/-6-3+2+-4+-2', '-8--8/-9+-4', '-6--2/-2-7*-3/4', '-9/4/-9/4+-6', '-2*5*7-3*3', '-1+-10/-9--4/-9*9', '10-4/-4/3--6', '9+-3/10', '9-1+-6--5+-10*-9', '8*1-3/3/-8*0', '-8/10+-7/6-5', '7-4+6*0/8/7', '6-3*7*-6', '-2/-9+5--8--5-5', '1/-7--2*-8+-4', '10-1*8--9', '-9/-8*-4+-7', '6/1/10/9/4-8', '10/9--7+-8/4', '-5-5*-2', '-5/-10--2+-7-6/-9', '9/-5*-8-10', '-9/6-1--8-5*5', '7/10-2', '6--2+-10/-5*10', '-10--4*-4*0-3', '7*-6+-10/7', '-2--3*9-8--7', '-7--6/10-8/10', '1+-7--8', '4*3*5+-4', '-10/-4/1-6', '-4*7--2+-6', '-3*-4*9*5-2', '-5--8-7', '10/-7*10-10', '4/2+-1', '-6-4/-5*-5', '-6+2--4-3/-5', '9-10*3*-3/-6/9', '2*4*8*-7-2', '-10+-6*4', '9/-3+-10+0-7/-5', '7-3+1+-5+3', '4-4*0+7', '1/5--2*8-7', '-1/-4+-4/6--8', '-1+5-8*5/2', '4--2+-9/-2', '0-5-8+-6-8', '7-2*5', '4*4-8', '-9+-7-8-9*7', '10-8+8*-10', '3*4*5-9--4+9', '3--1-2/-7/4/-3', '-4*-7+-3-10', '-5/3--7/6*-10+-6', '4*-4+-9/-9+3', '-4+-8*-4', '10/-2*-5-9-1/-10', '9/5+-8/-7+9', '6*7*3-3+-5+-9', '9-6*9', '1+-5/-2', '6+0-7*-7*-7', '3+-1*-3*-10--2', '1-9-7/-5+3+-3', '-3-0*9+-6', '9*-3+8-8+-10', '-4-9-6+9/-8', '8--9-2+10-10', '4+-5/10-7', '-7/-4*-5+-9', '4/-4/3-8', '1/5/10+2--6-0', '-8-10*9*-3/6', '1+-9/5*6--4', '5-10+-10/-5', '1/9+-7/-9*3', '-7+-9*-2--1', '3+-10/10-4-7', '10-4/-4-6*6', '-7*6-3-3', '-10+-5+-3*-9', '3-2+10-7', '7-3*-6*-8', '8+-10--1+5', '5--9-8-5-2', '-8+-5*0-9--7--4', '-7/-6+-4', '1*6+-6*-6', '-5/5/-2-10/-3--8', '7+6+-3*-3+6', '10--1-0+4', '9/-5+-4-10*9/-1', '8/4+-3*2', '-2--8+-4+-3', '1/2+-3', '-9/3-5+1+-4', '2*-8-9', '7/10+-9', '8-10/-6/-1--1/10', '-5/5-1+10', '10+-5-0/10/-2', '2/5+-10-3/-4+2', '-8-0+-7*-9+8--4', '-10/9+4+-7+5--8', '-2-2*-2-3-2', '-10*-7*4*-6+-5', '3--8*-1*10-1', '-2/8+3-5*-7', '-7/9*10-3', '7*3/2+-7/-3+10', '-10+3-10-8-7+4', '7--8-2*4', '0/-6*-6+-4*9', '1--1*9*5-6', '-10/-1-6*6-10--7', '10+2+-3-5/-10/6', '2*-7--1+-5', '1-10/-3/-7-9', '4*10/3*-7+-8', '-1/5-10/8/7', '-5+-9+-1/8', '0/-5*5+-7*-10', '9*0+-4--10', '1*4+-2--9', '-6-3+3', '6-5/-6*3-9', '-5+-6*6/-9-9', '9+-1--1--10*1', '5-8-1-2+-3', '6/9*5/-10-9', '1-4*-2/-3/-3*10', '-5/6-0+-8--9', '-6/2/7-8', '4*4+-6', '8--10*-8+-9--9', '4+-5-1*4', '-6+1+-3', '2/-9*-5-5', '7/4-1*3*-7', '4/3+-5', '7*3*8-7', '8+8-2/-1+6+-9', '-3-2--9', '-2/9-10*-4', '1+-4+-1*1', '5+-8/9+-4--2', '-2/-6*-4-0', '4/-10-8+-9--2', '-10/9+-3/1', '-10-2+4-5/-7', '-6-0/-10/6*3', '4/-9--1-7*-4', '6*5-10', '-1+-7/-10-5/-5', '-3--9*9*7+-1', '9+-6/9/-7', '-1+8-2*-7*3', '-8*1-8/7+2/-5', '-2-2/-7', '-3*2-8+-8', '-4+5-2*4--10+8', '0*-3+2+4-10', '4+4--6-7*-4*-5', '-10*3-2+1/-10*-7', '-2*-10+-3', '-3-5*0+-5+3', '7+6-2+-8', '-9*8/4-10+-1', '0-1/-7--1--2-10', '5*-2+-2*-7/-7', '-10*-4/5+-7/-1', '10--1+-6/7', '9/8+-2*-6/2', '6*5*-5-3*4-7', '-9+1+-2-3', '3/-5-7/-10-4-1', '-4+-10--8', '-1--4/-10+-5', '10+-1--8/-8/-4', '5+5/-6-1', '2/5--1+-7*4', '10/9-8-5*1', '3-4+4', '-10--8+-6', '7+-6+-4-7*-2', '4+-6-8', '-3-2/9+-4', '7+-4*9*-3/1', '-9*3+8+-8*5-9', '-1+-6--9', '-7*-3-9*9*5/-8', '-3--4--5-2+-2*-9', '-6--7--7-0-10', '-6*6+-7/4', '8*-3-6-0', '3-3/-10-2', '8-4/-7+9', '-2+0*-4+-7+5', '7/-10-3*-7--5--2', '-4+3--4-0', '6*-2--5+-10', '-1*-3+-4*10--8', '-9--1/2/-4+-10', '8-1+-7', '-10*-6+-6--5/-10', '-4-9*-2*1--4--5', '-7+-2/3--1', '-6-0+-4--2/-6', '-4+-5+-7-2--4*-5', '-8/8+-7*-7*-8', '-7-1*5--1+2-7', '3+3-3-7', '-6-1+0*5*4', '8--1-7+5+-7', '-1--8+-3/1', '8/-8+-10', '10*-10+-1/-9*10', '10+-3/5*-3*2', '-4--9+-8', '-3-9/6', '10/-10*-10-8', '2*0--7--1+-7/-4', '1--7-0/-4-9+-6', '-8+-9*8--9', '4*-9+-1+3', '4-4*1-10-4+7', '10*-7+-9--10*2', '6-0+-9+10/5', '1+-9+5/5', '6+-10*-8--7', '-9+-8-10--5', '-5*1-4/9*0', '4/-9-5*10+3', '6+-4+-1-2/-6', '7-2*-1+2', '4--8+-7--2*-9', '0+5/-3-3/6*6', '2-2/-5*-6+3', '-5*9+-4+6*-10*-5', '-10-0+9', '5+-1-2*-7*6', '4+5+-2', '3*-1+-3--3', '-8+1+10/-2+-6+-5', '-2+-5*7/-8', '3+8+0+-9+-7/5', '7+-7*-6+-8', '-4+-8/10*-3', '-6/1+-3-1', '-2+-3--3', '-7*-5*-4+-8-8+3', '10*4*-8+-8/10*-7', '9*3*8--1+-2*-4', '-8/2-5+10+-2', '10+0-0+8--2', '-7-6--9--10', '-9/5+-10--7', '9+-7*5*7', '6--1*0+-7', '-2*4-8-8', '5*-5+-4', '-4/-7+-5-6', '2+-1+-8+-9/4', '9/2-10--4', '5*-5-10/8/10*3', '-4*-2/10+1+-7*5', '7+-4*-8/-2', '-10+-2*2--5', '10/2/-8-10', '4-3/6-3', '-1+-10*-1', '-1--3-2/-7', '-3+2--4/7+6+-4', '-9*5-10*0-10/4', '10+-1/10+9', '5+2*0+7+-5', '-7/8*-8/-1-10', '-3-4*-4--9/-5', '-2+-7*8', '3+-9/-1-5', '-7/-8-7+10', '-9--10-6-7-6-0', '-1/5--8-9', '5-9-4/-4', '1+10-1-5', '-7--1+1-9/9', '8+-6*-9/3+-5', '6/8+-1', '-6+-6/6/1', '5/6/1*-3/-6-10', '2/-1*6-5--10', '8*10-10--3', '-6-1-9/8+-5', '-9/8/1/-6+-2/7', '1+1+-8', '-9/-7*6-9-1', '9+-7-8/10-6*5', '3-9+7*9', '-7/2-5+-4', '6-2+-6*3', '7/-6/-8--1--7+-10', '10-4/-4/-4+-2', '2+-4/3*-10', '6*5-2-7+8', '-8-9--1+2/-4', '-7+-9*-6*-9', '-2-6+-8+3', '5*-1/4-4', '8-2/1+1', '-6*-4+-9--10', '-8-10*1+-6', '-4-10-7/5', '1--2+9-10', '-3/-3+-4/9+6', '7+5/6-8+-3', '-7*8+7--10-9/3', '3+7*9-8--8', '9/-1+5-6', '5-4*9', '-4+3+4*0/5+-5', '-3+-10+2-1', '-6+7-6-7', '-5-9--8*-7--8*-8', '2/6+-1', '-9--4-1*-3/-1', '-4+-3+7/-9/-9-7', '8/2-9/4*3', '4-8+-9*-4+10+-2', '6+-6/4', '4+-3/4', '-3-5-6-9', '-4+10/10+-6/5/2', '8*2-0+-8', '1--4+-6--6*-9', '-8/-10+-8+-1', '6+-5/6--8-7/-4', '4+7*9-6', '9+-1*-1--4*4', '4*-5*-6+-3', '5/-3+-8+-6*-10-7', '8+-4+-3',
 '-1+4+3+-4*10--8', '6/-2+-5/7', '4*5+-3+6', '10-0*0--4*3--7', '6/3/-3+-4', '10-8*-7+4+-5*-10', '2-9-8+-10*-3+-9', '10*-6+3+-4+10', '9+1*2--2-8--5', '-5+-2+6-3/-10', '2-1/-6/9', '2+-5*10*1', '5*-9+-10+10', '0+1/2+-1*-1-1', '4*1/-3-9+7', '0/-2/4*8+-3', '8-5/-2', '2+9*-1/8-3', '2*10+-3+-8', '8+7/-7*-3+-9', '8--4*-5+5+-10+-3', '8+0+-9/10*-6', '-5*-6/3*4/-8-3', '-8--2*3-6/-10+-7', '1+-7*-9--7+-8', '-1+-4-8--1+-3', '8/5*1-1', '10/-6+-3', '-6*5-2/5', '6/-9/5*-3+-8', '-2/10+-7*5+6', '8+-5/-10--4', '-5+5/4+-1+-2', '-9*9/-6+-10', '-7/10+-9*10', '-4/3--1+6+-10', '1+-5*3', '-6+-1/4-6', '4-8-5*-6--10', '-2-10/9', '-5+-5-0--2', '-1+0+-3*-5/-8', '4+-9/3--5', '-10/-9*-9-10*5', '7*-6*4/2+-1', '1+0+-5+6', '-7/4+-10*-5--8', '-6--4/-4-3--9',
 '5+-2*6+7', '-1+3+-5*1', '6/-3*-9--7*-9+-6', '3+-3--3+8', '0+7/-5+-4-3', '8/-8+-4/7', '10*5+-2*1*-7', '-2+-2--8+8', '-7+-8--10*7*-1+2', '-10/4-3', '-4/3/-1+-9', '5/10-6+5/4', '-1-9*-7', '4+-1+-9*-10', '0/10+-4+6', '-1+8*2-0*9', '6-5+6--1-8', '-6*-5-2-0', '8--4--8+-6', '7+2-2--5/-5', '10--3*-10+-9/-10', '10/3-4+1+-8', '-4*-3+-9*9', '2/-6*-5-5/-10--9', '-4+-9+-10+-5', '-7+-1-10', '9/2-6', '4-7/1/8', '8*6-7-4--7/5', '7/-9+1+-1*8--9', '-4+-8--2+3/-8', '-10*0-10--8/5', '-1--8--1/9-4-10',
 '9+-2/-6+8+-8', '4+1+-2/10*9/5', '-10-8-9*9+6--5', '-5--8/-5+-5-1', '-1+-2+4/2', '1+5*8-7+0', '7-9-5', '2*-8/1+-5*-5', '-8*-9/-3/-1--3+-7', '-3*-3*4+-8+-9', '-9+-7/-5', '8-0+-10/-7*4+8', '2+-3*-5--1', '0+-8+-8*0', '5/-8+-9+10--10', '-3/-5-9-10', '8--10--9+-2', '7/10/-9+-9+8/7', '6/-6--10-8+-1*-1', '6/4-2']
