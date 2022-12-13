angleMode = 1  # 1 degrees,2 radians
functions = ['sin', 'cos', 'tan', 'acos', 'asin', 'atan', 'ln', 'log']

# precedence = {'+': 0, '-': 0, '*': 5, '/': 5, '%': 10, '^': 15}
precedence = {'+': 0, '-': 0, '*': 5, '/': 5, '%': 10, '^': 15}

for function in functions:
    precedence[function] = 20
    precedence['sin'] = 20
print(list(precedence.keys()))
