import os
import re
code = '''
class Calculate:

    def __init__(self, config: dict, vars=None):
        if vars == None:
            self.vars = {}
        else:
            self.vars = vars

    def computeExpression(self, expression):
        r = RPN(expression, self.vars)

        # print(r.rpn)
        return self.computeRPN(r)

    def computeRPN(self, RPN):
        paramStack = Stack()
        # print(f'computing {RPN.rpn}')
        print(f'rpn stack: {RPN.rpn}')
        for item in RPN.rpn:
            # print(item)

            if item not in precedence:
                paramStack.push(item)
            else:

                                    # need to computer whatever operator is in here
                # print(f'ITEM IS {item}')
                if item != 'sin' and item != 'cos' and item != 'tan':
                    x = paramStack.pop()
                    y = paramStack.pop()
                    print(f'x: {x}, y: {y}')
                    res = self.computeOp(x, y, item)
                    paramStack.push(res)
                else:
                    x = paramStack.pop()
                    res = self.computeTrig(item, x)
                    paramStack.push(res)

        return paramStack.pop()

    def findInput(self, op1):
        # gets value out of saved vars if it is not a number
        try:
            op1 = float(op1)
        except Exception:
            if op1 in self.vars:
                op1 = float(self.vars[op1])
            elif op1.find('x')!=-1:
                i=op1.find('x') 
                coef = float(op1[:i])
                x=self.findInput('x')
                op1=x*coef
            else:
                op1 = None  # error, the var isnt defined
        return op1

    def computeOp(self, op1, op2, symb):

        # print(f'computing {op2}{symb}{op1}')
        # try:
        #     op1 = float(op1)
        # except Exception:
        #     if op1 in self.vars:
        #         op1 = float(self.vars[op1])
        #     else:
        #         pass  # error, the var isnt defined
        op1 = self.findInput(op1)
        # try:
        #     op2 = float(op2)
        # except Exception:
        #     if op2 in self.vars:
        #         op2 = float(self.vars[op2])
        #     else:
        #         pass  # same error as before
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
            return math.sin(float(x))
        elif func == 'cos':
            return math.cos(float(x))
        else:
            return math.tan(float(x))

    def retrieveVariable(self, var):
        val = utils.readSetting(var)
        return val

'''
temp = code
code = code.replace('def ', 'SUBROUTINE ')

# code = re.sub("=.[a-z,A-Z]", "<--", code)
code = re.sub(" = ", "<--", code)

code = code.replace('==', '=')
with open('output.txt', 'w')as f:
    f.write(code)
