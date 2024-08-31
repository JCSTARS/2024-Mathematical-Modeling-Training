import math
import numpy as np
import pandas as pd
# 坡度
alpha = math.radians(1.5)
# 中央深度
h0 = 110
# 海域东西宽度
ew = 7408
# 海域南北宽度
sn = 3704

# 计算海域东边和西边边界的深度
deep_east = h0 - (ew / 2) * math.tan(alpha)
deep_west = h0 + (ew / 2) * math.tan(alpha)

def get_w1(h):
    # 深度为 h 处左边覆盖宽度
    return math.sin(math.radians(60)) * h / math.sin(math.radians(30 - 1.5))

def get_w2(h):
    # 深度为 h 处右边覆盖宽度
    return math.sin(math.radians(60)) * h / math.sin(math.radians(30 + 1.5))

def get_w(h):
    # 深度为 h 处覆盖宽度
    return get_w1(h) + get_w2(h)

def get_x(h):
    # 深度为 h 处距离西边海域边界的距离
    return (h0 + (ew / 2) * math.tan(alpha) - h) / math.tan(alpha)

# 初始化深度列表
deep = [deep_west]

# 循环计算每条测线的深度
i = 0
while get_x(deep[i]) + get_w2(deep[i]) * math.cos(alpha) < ew:
    # 计算新的深度值
    new_depth = deep_west - math.tan(alpha) * ((deep_west - ((get_x(deep[i]) + get_w2(deep[i]) * math.cos(alpha)) - (get_w(deep[i]) * math.cos(alpha) * 0.1)) * math.tan(alpha)) * math.sqrt(3) + ((get_x(deep[i]) + get_w2(deep[i]) * math.cos(alpha)) - (get_w(deep[i]) * math.cos(alpha) * 0.1)))
    deep.append(new_depth)
    i += 1

# 计算每条测线距离西边边界的距离
x_line = [get_x(d) for d in deep]

#print("每条测线处的海域深度:", deep)
#print("每条测线距离西边边界的距离:", x_line)

df1 = pd.DataFrame(deep, columns = ['Depth'])
df2 = pd.DataFrame(x_line, columns = ['Distance to the west border'])
df = pd.concat([df1, df2], axis=1)
df.to_excel('2023B/result3.xlsx', sheet_name='Sheet1', index=True)