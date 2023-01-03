import unittest
import re
from xml.etree.ElementTree import TreeBuilder
from utils import Stack
from config import precedence
from typing import List, Optional
# todo
# sort out the negatives DONE
# implement variable stored in the letters. just have them be the letters in rpn and deal with in compute
# add ability to implement functions sin cos tan asin acos atan dy/dx integrate ln log base
# create compute
# add short hand 5(6+4) and 6A


class RPN:
    def __init__(self, exp, variables):
        # Store the expression and variables in instance variables
        self.specialFuncs = ['(', 'sin', 'cos', 'tan', 'ln']

        self.variables = variables
        self.exp = exp
        self.precedence = precedence
        # Create a list of tokens (operators, parentheses, and variables)
        self.tokens = list(self.precedence.keys())
        self.tokens.append('(')
        self.tokens.append(')')
        # Convert the expression to RPN and store it in an instance variable
        self.rpn = self.CreateRPN(self.exp)

    def tokenise(self, exp: str) -> List[str]:
        '''
        This method takes an infix expression as input and returns a list of tokens(operators,
        parentheses, and variables).

        Args:
            exp(str): The infix expression to be tokenized.

        Returns:
            List[str]: A list of tokens representing the input expression.
        '''
        # Adapted from https://www.andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm
        # Remove spaces from the expression
        exp = exp.replace(' ', '')
        precedence = self.precedence
        tokens = self.tokens
        # print(f'TOKENS {tokens}')
        newExp = ''
        # \b(sin|cos|tan|ln|\()\b
        result = [_.start() for _ in re.finditer('sin|cos|tan|ln', exp)]

        # print(f'RESULT {result}')
        for res in result:
            exp = exp[:res]+' '+exp[res:]
        # result = re.split(r"\b(sin|cos|tan|ln|\(|\))\b", exp)
        # print('RESULT :', result)
        # newExp = ' '.join(result)

        # print(f'EXP {exp}')
        # Insert spaces before and after each token
        for token in tokens:
            for char in exp:
                if token == char:
                    # print(f'char: {char}')
                    # Space at the start in case the symbol follows from a number
                    newExp += f' {char} '
                else:
                    newExp += char  # It's a number

            exp = newExp  # ready to repeat for the next character

            newExp = ''
            # #print(f'newExp {exp}')
        # print(f'EXP {exp}')
        tokenised = exp.split(' ')  # do the split
        # print(f'TOKENISED FIRST PASS {tokenised}')
        newTokenised = []
        # remove the unecessary blank characters from subsequent symbols, ie ')*' ->' )  * ' which makes a mess when split
        for i, t in enumerate(tokenised):
            if t != '':
                newTokenised.append(t)
        tokenised = newTokenised
        # print('tokenised ', tokenised)

        newTokenised = []
        skip = False
        minus = False
        # Handle special cases with negative numbers
        if tokenised[0] == '-':
            tokenised.pop(0)
            t = tokenised.pop(0)
            tokenised.insert(0, f'-{t}')
        # print('1', tokenised)
        # Loop through the tokens

        for i, t in enumerate(tokenised[:-1]):
            # print(t, tokenised[i+1])
            if skip == True:
                # Skip this iteration

                skip = False
                continue
            # Check if this is a double minus (e.g. "--5")
            elif t == '-' and tokenised[i+1] == '-':
                # print(1)
                # Replace "--" with "+"

                newTokenised.append('+')
                skip = True
            # Check if we have a two operators together (e.g. "*-5")

            elif t in token and t != '+' and tokenised[i+1] == '-':
                # print(2)
                # print('in two ops together')
                newTokenised.append(t)
                skip = True
                minus = True
            # Check if we have a negative number

            elif t not in self.tokens and minus == True:
                # print(3)

                newTokenised.append(f'-{t}')
                minus = False
            else:
                # print(4)
                newTokenised.append(t)
            # print('2', newTokenised)
        # print('new tokenised ', newTokenised)
        # print('tokens ', self.tokens)
        # #print(len(tokenised))
        # Handle the special case of a minus at the end (e.g. "5 * 3 -")

        if newTokenised[-1] == '-' and newTokenised[-2] in self.tokens:
            newTokenised.pop()
            newTokenised.append(f'-{tokenised[-1]}')
        else:
            newTokenised.append(tokenised[-1])
        # print('new tokenised ', newTokenised)
        # print(f'FINAL TOKENS {newTokenised}')
        return newTokenised


# Adapted from https://www.andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm
    def CreateRPN(self, exp: str) -> List[str]:
        """
        Convert an infix expression to reverse polish notation (RPN).
        This method takes an infix expression as input and returns a list of tokens in RPN. This
        allows the expression to be easily evaluated using a stack.

        Args:
            exp(str): The infix expression to be converted to RPN.

        Returns:
            List[str]: A list of tokens representing the input expression in RPN.
        """
        specialFuncs = self.specialFuncs

        # Tokenize the expression
        tokenized = self.tokenise(exp)
        # print(f'FIRST IN TOKENISED {tokenized[0]}')
        # Create a stack for the shunting yard algorithm
        s = Stack()

        # Create a list for the output
        output = []

        # Loop through the tokens
        newTokenised = []
        for i, t in enumerate(tokenized):
            if t in specialFuncs and i != 0:
                # print(f'CONSIDERING IF {tokenized[i-1]}')
                # print(tokenized[i-1] not in ['+', '-', '*', '/'])
                if tokenized[i-1] not in ['+', '-', '*', '/'] and not(t == '(' and tokenized[i-1] in specialFuncs):
                    # s.push('*')
                    newTokenised.append('*')

            newTokenised.append(t)
            # print(f'NEW TOKENISED {newTokenised}')
        tokenized = newTokenised
        # print(f'tokenised {tokenized}')
        for i, t in enumerate(tokenized):

            if t == '(':

                # Push the left parenthesis onto the stack
                s.push(t)
            elif t == ')':
                # Pop all operators from the stack until we reach the left parenthesis
                while s.peek() != '(':
                    x = s.pop()
                    if x != '(':
                        output.append(x)
                # Pop the left parenthesis from the stack
                s.pop()
            elif t in self.tokens:
                # Pop all operators with greater or equal precedence from the stack
                while not s.isEmpty() and s.peek() in list(self.precedence.keys()):
                    if self.precedence[t] <= self.precedence[s.peek()]:
                        x = s.pop()
                        output.append(x)
                    else:
                        break
                # Push the current operator onto the stack
                s.push(t)
            else:
                # It's an operand, so add it to the output list
                output.append(t)

        # Pop all remaining operators from the stack and add them to the output
        while s.isEmpty() == False:
            x = s.pop()
            output.append(x)
        # print(f'RPN FOR FOR {self.exp} : {output}')
        return output

    def __repr__(self):
        return str(self.rpn)


'''
class RPN:
    def __init__(self, expression: str, vars: Optional[dict] = None):
        """Initialize the RPN class.

        Args:
            expression (str): The infix mathematical expression to convert to RPN.
            vars (dict): A dictionary of variables and their values.
        """
        self.expression = expression
        self.vars = vars
        self.rpn = self.convertToRPN()

    def convertToRPN(self) -> List[str]:
        """Convert the infix expression to RPN.

        Returns:
            The RPN representation of the infix expression.
        """
        rpn = []
        stack = []
        tokens = self.tokenize()
        skips = 0
        # Tokenize the expression
        for i, token in enumerate(tokens):
            if skips != 0:
                skips -= 1
                continue
            if token in self.vars:
                # rpn.append(str(self.vars[token]))
                if tokens[i+1] != '^':
                    # just shove a multiply there
                    rpn.append(token)
                    if tokens[i-1] not in precedence:
                        rpn.append('*')
                else:
                    rpn.append(token)
                    rpn.append('^')
                    rpn.append(tokens[i+2])
                    if tokens[i-1] not in precedence:

                        rpn.append('*')
                    skips = 2
            elif token.isdigit():
                rpn.append(token)
            elif token in precedence:
                while stack and precedence[token] <= precedence[stack[-1]]:
                    rpn.append(stack.pop())
                stack.append(token)
            elif token.isalpha():
                # Function call
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    rpn.append(stack.pop())
                stack.pop()
            else:
                # Invalid token
                raise ValueError(f"Invalid token: {token}")

        # Add remaining operators to the RPN list
        while stack:
            rpn.append(stack.pop())

        return rpn

    def tokenize(self) -> List[str]:
        """Tokenize the infix expression.

        Returns:
            A list of tokens in the infix expression.
        """
        tokens = []
        i = 0
        n = len(self.expression)

        # Iterate through the expression and extract individual tokens
        while i < n:
            if self.expression[i].isdigit():
                # Extract number
                num = ""
                while i < n and self.expression[i].isdigit():
                    num += self.expression[i]
                    i += 1
                tokens.append(num)
            elif self.expression[i].isalpha():
                # Extract function or variable
                ident = ""
                while i < n and self.expression[i].isalnum():
                    ident += self.expression[i]
                    i += 1
                tokens.append(ident)
            elif self.expression[i] in precedence:
                # Operator
                tokens.append(self.expression[i])
                i += 1
            else:
                # Parenthesis or invalid character
                tokens.append(self.expression[i])
                i += 1

        return tokens
'''

# class RPN:
#     def __init__(self, exp, variables):
#         # Store the expression and variables in instance variables
#         self.variables = variables
#         self.exp = exp
#         self.precedence = precedence
#         # Create a list of tokens (operators, parentheses, and variables)
#         self.tokens = list(self.precedence.keys())
#         self.tokens.append('(')
#         self.tokens.append(')')
#         # Convert the expression to RPN and store it in an instance variable
#         self.rpn = self.CreateRPN(self.exp)

#     def tokenise(self, exp):
#         # Adapted from https://www.andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm
#         # Remove spaces from the expression
#         exp = exp.replace(' ', '')
#         precedence = self.precedence
#         tokens = self.tokens
#         newExp = ''
#         # Insert spaces before and after each token
#         for token in tokens:
#             for char in exp:
#                 if char == token:
#                     # Space at the start in case the symbol follows from a number
#                     newExp += f' {char} '
#                 else:
#                     newExp += char  # It's a number

#             exp = newExp  # ready to repeat for the next character

#             newExp = ''
#             # print(f'newExp {exp}')

#         tokenised = exp.split(' ')  # do the split
#         newTokenised = []
#         # remove the unecessary blank characters from subsequent symbols, ie ')*' ->' )  * ' which makes a mess when split
#         for i, t in enumerate(tokenised):
#             if t != '':
#                 newTokenised.append(t)
#         tokenised = newTokenised

#         newTokenised = []
#         skip = False
#         minus = False
#         if tokenised[0] == '-':
#             tokenised.pop(0)
#             t = tokenised.pop(0)
#             tokenised.insert(0, f'-{t}')
#         for i, t in enumerate(tokenised[:-1]):
#             #print(t, tokenised[i+1])
#             if skip == True:
#                 skip = False
#                 continue
#             elif t == '-' and tokenised[i+1] == '-':
#                 newTokenised.append('+')
#                 skip = True
#             elif t in token and t != '+' and tokenised[i+1] == '-':
#                 print('in two ops together')
#                 newTokenised.append(t)
#                 skip = True
#                 minus = True

#             elif t not in self.tokens and minus == True:

#                 newTokenised.append(f'-{t}')
#                 minus = False
#             else:
#                 newTokenised.append(t)
#         # print(len(tokenised))
#         if newTokenised[-1] == '-' and newTokenised[-2] in self.tokens:
#             newTokenised.pop()
#             newTokenised.append(f'-{tokenised[-1]}')
#         else:
#             newTokenised.append(tokenised[-1])

#         return newTokenised

#     def convertToRPN(self) -> List[str]:
#         """Convert the infix expression to RPN.

#         Returns:
#             The RPN representation of the infix expression.
#         """
#         rpn = []
#         stack = []
#         tokens = self.tokenise(self.exp)
#         skips = 0
#         # Tokenize the expression
#         for i, token in enumerate(tokens):
#             if skips != 0:
#                 skips -= 1
#                 continue
#             if token in self.variables:
#                 # rpn.append(str(self.vars[token]))
#                 if tokens[i+1] != '^':
#                     # just shove a multiply there
#                     rpn.append(token)
#                     if tokens[i-1] not in precedence:
#                         rpn.append('*')
#                 else:
#                     rpn.append(token)
#                     rpn.append('^')
#                     rpn.append(tokens[i+2])
#                     if tokens[i-1] not in precedence:

#                         rpn.append('*')
#                     skips = 2
#             elif token.isdigit():
#                 rpn.append(token)
#             elif token in precedence:
#                 while stack and precedence[token] <= precedence[stack[-1]]:
#                     rpn.append(stack.pop())
#                 stack.append(token)
#             elif token.isalpha():
#                 # Function call
#                 stack.append(token)
#             elif token == '(':
#                 stack.append(token)
#             elif token == ')':
#                 while stack and stack[-1] != '(':
#                     rpn.append(stack.pop())
#                 stack.pop()
#             else:
#                 # Invalid token
#                 raise ValueError(f"Invalid token: {token}")

#         # Add remaining operators to the RPN list
#         while stack:
#             rpn.append(stack.pop())

#         return rpn

#     def CreateRPN(self, exp):
#         tokens = self.tokens
#         tokenised = self.tokenise(exp)
#         # shunting yard algorithm
#         s = Stack()
#         output = []
#         skips = 0
#         for i, t in enumerate(tokenised):
#             if skips != 0:
#                 skips -= 1
#                 continue
#             if t == '(':
#                 s.push(t)
#             elif t == ')':
#                 while s.peek() != '(':
#                     x = s.pop()
#                     if x != '(':
#                         output.append(x)
#                 s.pop()
#             elif t in tokens:
#                 while not s.isEmpty() and s.peek() in list(precedence.keys()):
#                     if precedence[t] <= precedence[s.peek()]:
#                         x = s.pop()
#                         output.append(x)
#                     else:
#                         break
#                 s.push(t)

#             else:
#                 # print('confirming x is inserted here ', t)
#                 if t.isdigit():

#                     output.append(t)
#                 else:
#                     if t in self.variables:
#                         # rpn.append(str(self.vars[token]))
#                         if tokens[i+1] != '^':
#                             # just shove a multiply there
#                             output.append(t)
#                             if tokens[i-1] not in precedence:
#                                 output.append('*')
#                         else:
#                             output.append(t)
#                             output.append('^')
#                             output.append(tokens[i+2])
#                             if tokens[i-1] not in precedence:

#                                 output.append('*')
#                             skips = 2
#         while s.isEmpty() == False:
#             x = s.pop()
#             output.append(x)

#         return output

#     def __repr__(self):
#         return str(self.rpn)


if __name__ == '__main__':
    r = RPN('3+9(x)^2+2', {'x': 1})
    print(r.rpn)
    r = RPN('3+9*x^2+1', {'x': 1})
    print(r.rpn)
    r = RPN('5*sin(6(x))', {'x': 1})
    print(r.rpn)
    # r = RPN('5*sin(6*x)', {'x': 1})
    # print(r.rpn)
