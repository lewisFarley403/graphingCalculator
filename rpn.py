import re
from xml.etree.ElementTree import TreeBuilder
from utils import Stack
from config import precedence
# todo
# sort out the negatives DONE
# implement variable stored in the letters. just have them be the letters in rpn and deal with in compute
# add ability to implement functions sin cos tan asin acos atan dy/dx integrate ln log base
# create compute
# add short hand 5(6+4) and 6A


class RPN:
    def __init__(self, exp, variables):
        self.variables = variables
        self.exp = exp
        self.precedence = precedence
        self.tokens = list(self.precedence.keys())
        self.tokens.append('(')
        self.tokens.append(')')
        self.rpn = self.CreateRPN(self.exp)

    def tokenise(self, exp):
        # adapted the algorithm from https://www.andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm
        exp = exp.replace(' ', '')
        precedence = self.precedence
        tokens = self.tokens
        newExp = ''
        # split on the spaces, 3+4-> 3 + 4 so that all the symbols and numbers can be tokenised
        for token in tokens:
            for char in exp:
                if char == token:
                    # space at the start incase the symbol follows from a number, numbers wont have spaces because theyre not in the token array
                    newExp += f' {char} '
                else:
                    newExp += char  # its a number
            exp = newExp  # ready to repeat for the next character

            newExp = ''
            # print(f'newExp {exp}')

        tokenised = exp.split(' ')  # do the split
        newTokenised = []
        # remove the unecessary blank characters from subsequent symbols, ie ')*' ->' )  * ' which makes a mess when split
        for i, t in enumerate(tokenised):
            if t != '':
                newTokenised.append(t)
        tokenised = newTokenised

        # sort out negative numbers

        # best if - not in precedence
        # # possible erroneous ouputs erroneous because cant cast to integer,
        # # a-b: compute replace with [a,-,b]
        # # a--b : replace with [a,+,b]
        # newTokens = []
        # for t in tokenised:
        #     print(t)
        #     if t != '-':
        #         # print(len(re.findall('-', t)))

        #         if len(re.findall('-', t)) == 0:
        #             # has nothing to do with negative numbers
        #             newTokens.append(t)
        #         elif len(re.findall('-', t)) == 1:

        #             split = t.split('-')

        #             # print(split)
        #             if split[0] != '':
        #                 newTokens.append(split[0])
        #                 newTokens.append('-')

        #                 newTokens.append(split[1])
        #             else:
        #                 # means its just a minus number ie -9
        #                 newTokens.append('-'+split[1])
        #         else:
        #             # -- is same as plus
        #             split = t.split('--')
        #             # print(split)

        #             newTokens.append(split[0])
        #             newTokens.append('+')

        #             newTokens.append(split[1])

        # - in pres
        # idea: alter what the minuses mean

        # case one: [a,-,b]: leave it
        # case 2 [-,-,a]: +a
        # [-,a,SYMBOL]: [-a,SYMBOL] because

        newTokenised = []
        skip = False
        minus = False
        if tokenised[0] == '-':
            tokenised.pop(0)
            t = tokenised.pop(0)
            tokenised.insert(0, f'-{t}')
        for i, t in enumerate(tokenised[:-1]):
            #print(t, tokenised[i+1])
            if skip == True:
                skip = False
                continue
            elif t == '-' and tokenised[i+1] == '-':
                newTokenised.append('+')
                skip = True
            elif t in token and t != '+' and tokenised[i+1] == '-':
                print('in two ops together')
                newTokenised.append(t)
                skip = True
                minus = True

            elif t not in self.tokens and minus == True:

                newTokenised.append(f'-{t}')
                minus = False
            else:
                newTokenised.append(t)
        # print(len(tokenised))
        if newTokenised[-1] == '-' and newTokenised[-2] in self.tokens:
            newTokenised.pop()
            newTokenised.append(f'-{tokenised[-1]}')
        else:
            newTokenised.append(tokenised[-1])

        return newTokenised

    def CreateRPN(self, exp):
        tokens = self.tokens
        tokenised = self.tokenise(exp)
        # shunting yard algorithm
        s = Stack()
        output = []
        for i, t in enumerate(tokenised):
            if t == '(':
                s.push(t)
            elif t == ')':
                while s.peek() != '(':
                    x = s.pop()
                    if x != '(':
                        output.append(x)
                s.pop()
            elif t in tokens:
                while not s.isEmpty() and s.peek() in list(precedence.keys()):
                    if precedence[t] <= precedence[s.peek()]:
                        x = s.pop()
                        output.append(x)
                    else:
                        break
                s.push(t)

            else:
                output.append(t)
        while s.isEmpty() == False:
            x = s.pop()
            output.append(x)

        return output

    def __repr__(self):
        return str(self.rpn)


if __name__ == '__main__':
    #r = RPN('A-6', {})
    r = RPN('5*tan (6+5)', {})
    print(r.tokenise(r.exp))
    #r = RPN('5-6*-9', {})
    # print('r: ', r)
    # r = RPN('-6+4', {})

    # print('r: ',r)
    test1 = RPN('5-6*-9', {})
    test2 = RPN('-10+9--5+-6', {})
    test3 = RPN('10-1', {})
    print(test1.tokenise('5-6*-9'))
    print(test2.tokenise(test2.exp))
    print(test3.tokenise(test3.exp))
