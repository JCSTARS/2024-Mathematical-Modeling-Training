import math
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

theta = math.radians(120)
alpha = math.radians(-1.5)
htheta = theta / 2 
h = 120  
unit = 1852  
k0 = np.tan(alpha) 

W = np.zeros((9, 9))
for i in range(1,9):
    W[0][i] = (i-1) * 0.3
    W[i][0] = (i-1) * 45
for i in range(1, 9):
    for j in range(1, 9):
        beta = (i - 1) * np.pi / 4
        d = (j - 1) * 0.3 * unit  # 换算为米
        v = np.array([np.cos(beta), np.sin(beta), 0])  # 直线的法向量
        origin = v * d  
        
        # 波束的方向向量
        v1 = np.array([-np.sin(beta) * np.sin(htheta), np.cos(beta) * np.sin(htheta), -np.cos(htheta)])
        v2 = np.array([np.sin(beta) * np.sin(htheta), -np.cos(beta) * np.sin(htheta), -np.cos(htheta)])
        
        leftsolve = lambda t: (v1[0] * t + origin[0]) * k0 - h - (v1[2] * t + origin[2])
        rightsolve = lambda t: (v2[0] * t + origin[0]) * k0 - h - (v2[2] * t + origin[2])
        
        # 解方程找到交点
        tleft = fsolve(leftsolve, 0)
        tright = fsolve(rightsolve, 0)
        
        # 计算左右交点的坐标
        pleft = v1 * tleft + origin
        pright = v2 * tright + origin
        
        # 计算覆盖宽度
        W[i, j] = np.linalg.norm(pleft - pright)

df = pd.DataFrame(W)
df.to_excel('2023B/result2.xlsx', sheet_name='Sheet1', index=False, header=False)