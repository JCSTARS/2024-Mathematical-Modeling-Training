import math
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

L = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
D = np.zeros(9)
W = np.zeros(9)
Y = np.zeros(9)
x1 = np.zeros(9)
x2 = np.zeros(9)

def func1(x):
    t = math.radians(60)
    a = math.radians(1.5)
    return D[i] + (math.tan(a) - 1/math.tan(t)) * x

def func1(x):
    t = math.radians(60)
    a = math.radians(1.5)
    return D[i] - (math.tan(a) + 1/math.tan(t)) * x

alpha = 1.5
theta = 120
alpha = math.radians(alpha)
theta = math.radians(theta)
for i in range(9):
    D[i] = 70 - L[i] * math.sin(alpha)
for i in range(9):
    x1[i] = fsolve(func1, D[i]*math.tan(theta/2) - 10)
    x2[i] = fsolve(func1, D[i]*math.tan(theta/2) + 10)
    W[i] = x1[i] + x2[i]
    if i > 0:
        Y[i] = (x2[i] + x1[i-1] - 200)/W[i-1]
        # 当覆盖率<0时，认为覆盖率=0
        if Y[i] < 0:
            Y[i] = 0
df1 = pd.DataFrame(L, columns = ['Distance'])
df2 = pd.DataFrame(D, columns = ['Depth'])
df3 = pd.DataFrame(W, columns = ['Coverage width'])
df4 = pd.DataFrame(Y*100, columns = ['Overlap'])
df = pd.concat([df1, df2, df3, df4], axis=1)
df.to_excel('2023B/result1.xlsx', sheet_name='Sheet1', index=False)