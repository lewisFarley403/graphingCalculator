from decimal import Decimal
from math import log,e
def func(x):
    return Decimal(log(x,e))
def differnetiate(function,x):
    h=Decimal(1e-10)
    diff = function(x+h)-function(x)
    return Decimal(diff/h)
print(differnetiate(func,2))