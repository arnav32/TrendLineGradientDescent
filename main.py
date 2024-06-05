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

# --------------- Differentiate with respect to INTERCEPT ---------------

# d/d_intercept of (observed - ((slope * x_val) + intercept)) ** 2
def deriv_squared_res_intercept(x_val, observed, slope):
    intercept = symbols('intercept', real=True) # intercept: intercept
    f = (observed - ((slope * x_val) + intercept)) ** 2
    return(diff(f, intercept))

def deriv_sum_sqared_res_intercept(x_vals, observeds, slope):
    sum_derivs = []
    for i in range(len(x_vals)):
        sum_derivs.append(deriv_squared_res_intercept(x_vals[i], observeds[i], slope))
    return sum_derivs

# --------------- Differentiate with respect to SLOPE ---------------

# d/d_intercept of (observed - ((slope * x_val) + intercept)) ** 2
def deriv_squared_res_slope(x_val, observed, intercept):
    slope = symbols('slope', real=True)
    f = (observed - ((slope * x_val) + intercept)) ** 2
    return(diff(f, slope))

def deriv_sum_sqared_res_slope(x_vals, observeds, intercept):
    sum_derivs = []
    for i in range(len(x_vals)):
        sum_derivs.append(deriv_squared_res_slope(x_vals[i], observeds[i], intercept))
    return sum_derivs

# loss function
def sum_squared_res(x_vals, observeds, intercept, slope):
    sum_squared_res = 0
    # for x_val, observed in x_vals, observeds:
    for i in range(len(x_vals)):
        sum_squared_res += squared_res(x_vals[i], observeds[i], intercept, slope)
    return sum_squared_res



# def plot_graph(x_vals, observeds, intercept, slope):
    # fig = plt.figure(figsize=(5.5,5.5))
    # plt.ion()
    # x = np.linspace(min(x_vals), max(x_vals), 5) # takes: start, stop, number of vals
    # ax = fig.add_subplot(111)
    # # ax.set_ylim([0, max(observeds) * 1.2])
    # ax.set_ylim(ymin=min(observeds) * 0.8, ymax=(max(observeds) * 1.2))
    # m = slope
    # c = intercept
    # y = m*x+c
    # line1, = ax.plot(x, y, 'r-')
    # line1.set_ydata((slope * x) + intercept)

    # # plt.figure(figsize=(3.5,3.5))
    # # plt.plot(x, m*x+c, linestyle='solid')

    # # for i in range(len(x_vals)):
    # #     plt.plot(x_vals[i], observeds[i], 'bo')

    # xpoints = np.array(x_vals)
    # ypoints = np.array(observeds)

    # plt.plot(xpoints, ypoints, 'bo')



    
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # # plt.show()
    # plt.draw()
    # sleep(0.1)
    
    # # plt.close()



def calc_slope_sq_res_intercept(x_vals, observeds, slope, intercept):
    sum = 0
    expression = deriv_sum_sqared_res_intercept(x_vals, observeds, slope)
    for term in expression:
        term = str(term)
        # print(term)
        term = term.replace("intercept", str(intercept))
        # print(term)
        # sum += eval_expr(term)
        sum += nsp.eval(term)
        # print(str(sum) + "\n")
    return sum


def calc_slope_sq_res_slope(x_vals, observeds, slope, intercept):
    sum = 0
    expression = deriv_sum_sqared_res_slope(x_vals, observeds, intercept)
    for term in expression:
        term = str(term)
        # print(term)
        term = term.replace("slope", str(slope))
        # print(term)
        # sum += eval_expr(term)
        sum += nsp.eval(term)
        # print(str(sum) + "\n")
    return sum

def calc_new_intercept(old_intercept, learning_rate, slope_sq_res_intercept):
    step_size_intercept = slope_sq_res_intercept * learning_rate
    new_intercept = old_intercept - step_size_intercept
    return new_intercept

def calc_new_slope(old_slope, learning_rate, slope_sq_res_slope):
    step_size_slope = slope_sq_res_slope * learning_rate
    new_slope = old_slope - step_size_slope
    return new_slope



# =================== MAIN PROGRAM ===================

observed_points = [(0.5, 6.4), 
                   (2.3, 6.9), 
                   (2.9, 8.2), 
                   (4.6, 14.5), 
                   (6.1, 16.7)
                  ]

x_vals, observeds = zip(*observed_points)
intercept = 8
slope = 6

# Choose parameters
learning_rate = 0.01
max_steps = 20
sleep_time = 1


step_size_intercept = sys.maxsize
step_size_slope = sys.maxsize

# plot_graph(x_vals, observeds, intercept, slope)
sleep(2)

clear = lambda: print("\033c", end="", flush=True)
clear()

# for i in range(10):
steps_num = 0






# Initialise plot
plt.ion()
# def plot_graph(x_vals, observeds, intercept, slope):
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111)
# ax.set_ylim([0, max(observeds) * 1.2])
# ax.set_ylim(ymin=min(observeds) * 0.8, ymax=(max(observeds) * 1.2))
ax.set_ylim(ymin=0, ymax=(max(observeds) * 1.2))


# -------------- Draw initial graph --------------
# Plot observed values as blue points
xpoints = np.array(x_vals)
ypoints = np.array(observeds)
plt.plot(xpoints, ypoints, 'bx', markersize=8)

# Plot line
line2 = plt.axline((0, intercept), slope=slope, linewidth=1.5, color='r')

# Draw graph
fig.canvas.draw()
fig.canvas.flush_events()
# plt.show()
plt.draw()
sleep(sleep_time)
# line1.remove()
line2.remove()
steps_num += 1
# clear()

# for i in range(2):
while abs(step_size_intercept) > 0.001 and steps_num <= max_steps:
    print(f"--------------------- Step number {steps_num} ---------------------\n")
    print(f"Previous intercept: {intercept}")
    print(f"Previous slope: {slope}")
    print(f"\nPrevious loss function: {round(sum_squared_res(x_vals, observeds, intercept, slope), 4)}\n")

    # -------- Print all values --------

        # -------- INTERCEPT --------
    slope_sq_res_intercept = calc_slope_sq_res_intercept(x_vals, observeds, slope, intercept)
    # print(f"Slope of sq res w respect to INTERCEPT: {slope_sq_res_intercept}")
    print(f"Derivative of loss function w.r. to INTERCEPT: {round(slope_sq_res_intercept, 4)}")
    
    step_size_intercept = slope_sq_res_intercept * learning_rate
    print(f"Step size for INTERCEPT: {round(step_size_intercept, 4)}\n")
    
    intercept = calc_new_intercept(intercept, learning_rate, slope_sq_res_intercept)

        # ---------- SLOPE ----------
    slope_sq_res_slope = calc_slope_sq_res_slope(x_vals, observeds, slope, intercept)
    print(f"Derivative of loss function w.r. to SLOPE: {round(slope_sq_res_slope, 4)}")

    step_size_slope = slope_sq_res_slope * learning_rate
    print(f"Step size for SLOPE: {round(step_size_slope, 4)}\n")

    slope = calc_new_slope(slope, learning_rate, slope_sq_res_slope)


    print(f"New intercept: {intercept}")
    print(f"New slope: {slope}\n\n") 

    # -------- Draw updated graph --------

    # Plot observed values as blue points
    xpoints = np.array(x_vals)
    ypoints = np.array(observeds)
    plt.plot(xpoints, ypoints, 'bx', markersize=8)

    # Plot line
    line2 = plt.axline((0, intercept), slope=slope, linewidth=1.5, color='r')

    # Draw graph
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.show()
    plt.draw()

    sleep(sleep_time)
    # line1.remove()
    line2.remove()
    steps_num += 1
    # clear()


# -------- Draw final graph --------

# Plot observed values as blue points
xpoints = np.array(x_vals)
ypoints = np.array(observeds)
plt.plot(xpoints, ypoints, 'bx', markersize=8)

# Plot line
line2 = plt.axline((0, intercept), slope=slope, linewidth=1.5, color='r')

# Draw graph and keep it
fig.canvas.draw()
plt.draw()
plt.show(block=True)






# x = np.linspace(min(x_vals), max(x_vals), 5) # takes: start, stop, number of vals
# m = slope
# c = intercept
# y = m*x+c
# line1, = ax.plot(x, y, 'r-')
# line1.set_ydata((slope * x) + intercept)

# # plt.figure(figsize=(3.5,3.5))
# # plt.plot(x, m*x+c, linestyle='solid')

# # for i in range(len(x_vals)):
# #     plt.plot(x_vals[i], observeds[i], 'bo')

# xpoints = np.array(x_vals)
# ypoints = np.array(observeds)

# plt.plot(xpoints, ypoints, 'bo')

# fig.canvas.draw()
# # plt.show()
# plt.draw()
# plt.show(block=True)






# ------------------------ NOTES ------------------------
# for i in range(20): # TODO: change this loop condition later
#     slope = derviative_intercept(ssq)
#     step_size_intercept = slope * learning_rate



