class RPN:
    def __init__(self, expression: str, vars: Optional[dict] = None):
        """Convert an infix mathematical expression to reverse Polish notation.

        Args:
            expression (str): The infix mathematical expression to convert.
            vars (dict): A dictionary of variables and their values.
        """
        self.rpn = []  # the converted RPN expression
        self.opStack = []  # stack for operators
        self.vars = vars  # letters that have a value attatched to them

        # split the input into tokens
        tokens = self.tokenize(expression)

        # convert the tokens to RPN
        self.convertToRPN(tokens)

    def tokenize(self, expression: str) -> list:
        """Split an infix expression into tokens.

        Args:
            expression (str): The infix expression to split into tokens.

        Returns:
            A list of tokens in the expression.
        """
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i] in '+-*/^()':
                tokens.append(expression[i])
                i += 1
            elif expression[i] in '0123456789.':
                num = ''
                while i < len(expression) and expression[i] in '0123456789.':
                    num += expression[i]
                    i += 1
                tokens.append(num)
            else:  # must be a variable
                var = ''
                while i < len(expression) and expression[i] not in '+-*/^()':
                    var += expression[i]
                    i += 1
                tokens.append(var)
        return tokens

    def convertToRPN(self, tokens: list):
        """Convert a list of tokens from infix to reverse Polish notation.

        Args:
            tokens (list): The list of tokens in infix notation.
        """
        for token in tokens:
            if token not in precedence:
                self.rpn.append(token)
            elif token == '(':
                self.opStack.append(token)
            elif token == ')':
                while self.opStack and self.opStack[-1] != '(':
                    self.rpn.append(self.opStack.pop())
                self.opStack.pop()  # remove the '('
            else:  # must be an operator
                while self.opStack and self.opStack[-1] != '(' and precedence[token] <= precedence[self.opStack[-1]]:
                    self.rpn.append(self.opStack.pop())
                self.opStack.append(token)
        while self.opStack:
            self.rpn.append(self.opStack.pop())
