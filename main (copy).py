import ast
import operator as op
import sys

import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff
from time import sleep

from eval import NumericStringParser

nsp = NumericStringParser()

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)

def eval_(node):
    match node:
        case ast.Constant(value) if isinstance(value, int):
            return value  # integer
        # case ast.BinOp(left, op, right):
        #     return operators[type(op)](eval_(left), eval_(right))
        # case ast.UnaryOp(op, operand):  # e.g., -1
        #     return operators[type(op)](eval_(operand))
        case _:
            raise TypeError(node)


def squared_res(x_val, observed, intercept, slope):
    predicted = (slope * x_val) + intercept
    return (observed - predicted) ** 2

# d/d_intercept of (observed - ((slope * x_val) + intercept)) ** 2
def deriv_squared_res(x_val, observed, slope):
    intercept = symbols('intercept', real=True) # intercept: intercept
    f = (observed - ((slope * x_val) + intercept)) ** 2
    return(diff(f, intercept))

def deriv_sum_sqared_res(x_vals, observeds, slope):
    sum_derivs = []
    for i in range(len(x_vals)):
        sum_derivs.append(deriv_squared_res(x_vals[i], observeds[i], slope))
    return sum_derivs

# loss function
def sum_squared_res(x_vals, observeds, intercept, slope):
    sum_squared_res = 0
    # for x_val, observed in x_vals, observeds:
    for i in range(len(x_vals)):
        sum_squared_res += squared_res(x_vals[i], observeds[i], intercept, slope)
    return sum_squared_res

def plot_graph(x_vals, observeds, intercept, slope):
    x = np.linspace(min(x_vals), max(x_vals), 5) # takes: start, stop, number of vals
    m = slope
    c = intercept
    y = m*x+c
    plt.ion() 
    fig = plt.figure(figsize=(3.5,3.5)) 
    ax = fig.add_subplot(111) 
    # ax.set_ylim([0, max(observeds) * 1.2])
    ax.set_ylim(ymin=min(observeds) * 0.8, ymax=(max(observeds) * 1.2))
    line1, = ax.plot(x, y, 'b-') 
    
    # plt.figure(figsize=(3.5,3.5))
    # plt.plot(x, m*x+c, linestyle='solid')

    line1.set_ydata((slope * x) + intercept)
    
    
    for i in range(len(x_vals)):
        plt.plot(x_vals[i], observeds[i], 'bo')
    fig.canvas.draw() 
    plt.show()
    fig.canvas.flush_events() 

def calc_slope(x_vals, observeds, slope, intercept):
    sum = 0
    expression = deriv_sum_sqared_res(x_vals, observeds, slope)
    for term in expression:
        term = str(term)
        # print(term)
        term = term.replace("intercept", str(intercept))
        # print(term)



        # sum += eval_expr(term)
        sum += nsp.eval(term)




        # print(str(sum) + "\n")
    return sum

def calc_new_intercept(old_intercept, learning_rate, slope_sq_res):
    step_size = slope_sq_res * learning_rate
    new_intercept = old_intercept - step_size
    return new_intercept

x_vals = [0.5, 2.3, 2.9, 4.6, 6.1]
observeds = [6.4, 6.9, 8.2, 14.5, 16.7]
intercept = 10
slope = 1.8

learning_rate = 0.1



# plot_graph(x_vals, observeds, intercept, slope))
# print(squared_res(6, 2, 4, 8))
# print(deriv_squared_res(1.4, 0.5, 0.64))
# print(deriv_sum_sqared_res(x_vals, observeds, slope))

step_size = sys.maxsize

plot_graph(x_vals, observeds, intercept, slope)
sleep(2)

# for i in range(10):
while abs(step_size) > 0.001:
    print(f"Current intercept: {intercept}")
    print(f"Loss function: {sum_squared_res(x_vals, observeds, intercept, slope)}")
    slope_sq_res = calc_slope(x_vals, observeds, slope, intercept)
    print(f"Slope of squared residuals: {slope_sq_res}")
    step_size = slope_sq_res * learning_rate
    print(f"Step size: {step_size}")
    intercept = calc_new_intercept(intercept, learning_rate, slope_sq_res)
    print(f"New intercept: {intercept}\n\n")
    
    plot_graph(x_vals, observeds, intercept, slope)
    sleep(2)


# for i in range(20): # TODO: change this loop condition later
#     slope = derviative_intercept(ssq)
#     step_size = slope * learning_rate