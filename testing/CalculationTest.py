import random
from functionalityPages import Calculate
precedence = {'+': 0, '*': 5,'/':0,'-':0}
NUMBER_OF_TESTS =1000
P_OF_BRACKETS = 0.0
expressions = []
def createBracketExp():
    n=random.randint(5,10)
    bracketSec =''
    locNumber = True
    for i in range(n):
        r = random.uniform(0,1)
        if locNumber == True:
            if r<P_OF_BRACKETS:
                sec =createBracketExp()
                bracketSec+=f'({sec})'
                
            else:
                bracketSec+=str(random.randint(-10,10))
        else:
            bracketSec+=random.choice(list(precedence.keys()))
        locNumber = not locNumber
    if locNumber == True:
        bracketSec+=str(random.randint(-10,10))
    return bracketSec


for _ in range(NUMBER_OF_TESTS):
#     chars = random.randint(5,100)
#     number = True
#     exp =''
    expressions.append(createBracketExp())
failList = []
x=eval('9*5-4*(9*3-3/1*5)+9')
c=Calculate({})
# print(c.computeExpression('9*5-4*(9*3-3/1*5)+9'),x)
failed =0
for ex in expressions:
    print('EXPRESSION ',ex)
    try:
        
        x=eval(ex)
        y=c.computeExpression(ex)
    except ZeroDivisionError:
        continue
    

    if x!=y:
        failed +=1
        print(f'{ex} failed the test case. Eval got {x} i got {y}')
        failList.append(ex)
    else:
        print('passed test case')
print(failed)
print(failList)