'''
def parse_expression():
    result = parse_term()
    while True:
        if match('+'):
            result += parse_term()
        elif match('-'):
            result -= parse_term()
        else:
            return result


def parse_term():
    result = parse_number()
    while True:
        if match('*'):
            result *= parse_number()
        elif match('/'):
            result /= parse_number()
        else:
            return result


def parse_number():
    if match('-'):
        return -parse_number()
    result = 0
    while match_any(string.digits):
        result = result * 10 + int(next())
    return result


# Set the input string
input_str = "2 + 3 * 4"

# Parse and evaluate the expression
result = parse_expression()

print(result)  # Output: 14
'''


import re


def shunting_yard(expression):
    output = []
    stack = []
    last_token = None
    for token in expression:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif last_token and last_token.isdigit() and token not in ('(', ')'):
            # Handle implicit multiplication
            output.append('*')
            stack.append(token)
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif last_token and last_token.isdigit() and token not in ('(', ')'):
            # Handle implicit multiplication
            output.append('*')
            stack.append(token)
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a number and an opening parenthesis
                output.append('*')
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a number and an opening parenthesis
                output.append('*')
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a closing parenthesis and a number
                stack.append('*')
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a number and an opening parenthesis
                output.append('*')
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a closing parenthesis and a number
                stack.append('*')
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a number and an opening parenthesis
                output.append('*')
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a closing parenthesis and a number
                output.append('*')
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
    while stack:
        output.append(stack.pop())
    return output


def shunting_yard(expression):
    # Split the input string into a list of tokens
    tokens = re.findall(r'\d+|[^\d\s]+', expression)

    output = []
    stack = []
    last_token = None
    for token in tokens:
        if token.isdigit():
            stack.append(token)
        elif token == '(':
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a number and an opening parenthesis
                output.append('*')
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
            if last_token and last_token.isdigit():
                # Insert a multiplication operator between a closing parenthesis and a number
                output.append('*')
        else:
            while stack and precedence(token) <= precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(token)
        last_token = token
        print(last_token)
    while stack:
        output.append(stack.pop())
    return output


def precedence(token):
    if token in ('+', '-'):
        return 1
    elif token in ('*', '/'):
        return 2
    elif token == '^':
        return 3
    else:
        return 0


def evaluate_rpn(expression):
    stack = []
    for token in expression:
        if token.isdigit():
            stack.append(int(token))
        else:
            # Pop the last two operands from the stack
            operand_2 = stack.pop()
            operand_1 = stack.pop()
            # Perform the operation and push the result back onto the stack
            if token == '+':
                result = operand_1 + operand_2
            elif token == '-':
                result = operand_1 - operand_2
            elif token == '*':
                result = operand_1 * operand_2
            elif token == '/':
                result = operand_1 / operand_2
            elif token == '^':
                result = operand_1 ** operand_2
            stack.append(result)
    # The final result should be the only value left on the stack
    return stack[0]


# Convert the infix expression "5(10*8+6)" to RPN
expression = "5(10*8+6)"
output = shunting_yard(expression)
print(output)  # Output:
print(evaluate_rpn(['5', '6', 'x', '*', 'sin', '*']))
