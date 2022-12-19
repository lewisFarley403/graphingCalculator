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

#     def tokenise(self, exp: str) -> List[str]:
#         '''
#         This method takes an infix expression as input and returns a list of tokens(operators,
#         parentheses, and variables).

#         Args:
#             exp(str): The infix expression to be tokenized.

#         Returns:
#             List[str]: A list of tokens representing the input expression.
#         '''
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
#             # #print(f'newExp {exp}')

#         tokenised = exp.split(' ')  # do the split
#         newTokenised = []
#         # remove the unecessary blank characters from subsequent symbols, ie ')*' ->' )  * ' which makes a mess when split
#         for i, t in enumerate(tokenised):
#             if t != '':
#                 newTokenised.append(t)
#         tokenised = newTokenised
#         #print('tokenised ', tokenised)

#         newTokenised = []
#         skip = False
#         minus = False
#         # Handle special cases with negative numbers
#         if tokenised[0] == '-':
#             tokenised.pop(0)
#             t = tokenised.pop(0)
#             tokenised.insert(0, f'-{t}')
#         #print('1', tokenised)
#         # Loop through the tokens

#         for i, t in enumerate(tokenised[:-1]):
#             ##print(t, tokenised[i+1])
#             if skip == True:
#                 # Skip this iteration

#                 skip = False
#                 continue
#             # Check if this is a double minus (e.g. "--5")
#             elif t == '-' and tokenised[i+1] == '-':
#                 # print(1)
#                 # Replace "--" with "+"

#                 newTokenised.append('+')
#                 skip = True
#             # Check if we have a two operators together (e.g. "*-5")

#             elif t in token and t != '+' and tokenised[i+1] == '-':
#                 # print(2)
#                 #print('in two ops together')
#                 newTokenised.append(t)
#                 skip = True
#                 minus = True
#             # Check if we have a negative number

#             elif t not in self.tokens and minus == True:
#                 # print(3)

#                 newTokenised.append(f'-{t}')
#                 minus = False
#             else:
#                 # print(4)
#                 newTokenised.append(t)
#             #print('2', newTokenised)
#         #print('new tokenised ', newTokenised)
#         #print('tokens ', self.tokens)
#         # #print(len(tokenised))
#         # Handle the special case of a minus at the end (e.g. "5 * 3 -")

#         if newTokenised[-1] == '-' and newTokenised[-2] in self.tokens:
#             newTokenised.pop()
#             newTokenised.append(f'-{tokenised[-1]}')
#         else:
#             newTokenised.append(tokenised[-1])
#         #print('new tokenised ', newTokenised)
#         return newTokenised


# # Adapted from https://www.andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm
#     def CreateRPN(self, exp: str) -> List[str]:
#         """
#         Convert an infix expression to reverse polish notation (RPN).
#         This method takes an infix expression as input and returns a list of tokens in RPN. This
#         allows the expression to be easily evaluated using a stack.

#         Args:
#             exp(str): The infix expression to be converted to RPN.

#         Returns:
#             List[str]: A list of tokens representing the input expression in RPN.
#         """

#         # Tokenize the expression
#         tokenized = self.tokenise(exp)

#         # Create a stack for the shunting yard algorithm
#         s = Stack()

#         # Create a list for the output
#         output = []

#         # Loop through the tokens
#         for t in tokenized:
#             if t == '(':
#                 # Push the left parenthesis onto the stack
#                 s.push(t)
#             elif t == ')':
#                 # Pop all operators from the stack until we reach the left parenthesis
#                 while s.peek() != '(':
#                     x = s.pop()
#                     if x != '(':
#                         output.append(x)
#                 # Pop the left parenthesis from the stack
#                 s.pop()
#             elif t in self.tokens:
#                 # Pop all operators with greater or equal precedence from the stack
#                 while not s.isEmpty() and s.peek() in list(self.precedence.keys()):
#                     if self.precedence[t] <= self.precedence[s.peek()]:
#                         x = s.pop()
#                         output.append(x)
#                     else:
#                         break
#                 # Push the current operator onto the stack
#                 s.push(t)
#             else:
#                 # It's an operand, so add it to the output list
#                 output.append(t)

#         # Pop all remaining operators from the stack and add them to the output
#         while s.isEmpty() == False:
#             x = s.pop()
#             output.append(x)

#         return output

#     def __repr__(self):
#         return str(self.rpn)


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

        # Tokenize the expression
        for token in tokens:
            if token in self.vars:
                rpn.append(str(self.vars[token]))
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



if __name__ == '__main__':
    '''
    # r = RPN('A-6', {})
    r = RPN('5*tan (6+5)', {})
    #print(r.tokenise(r.exp))
    # r = RPN('5-6*-9', {})
    # #print('r: ', r)
    # r = RPN('-6+4', {})

    # #print('r: ',r)
    test1 = RPN('5-6*-9', {})
    test2 = RPN('-10+9--5+-6', {})
    test3 = RPN('10-1', {})
    #print(test1.tokenise('5-6*-9'))
    #print(test2.tokenise(test2.exp))
    #print(test3.tokenise(test3.exp))
'''
