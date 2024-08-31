import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
def calc_angle(x1,y1,x2,y2,x3,y3):
    # 给定点A, B, C的坐标
    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)

# 计算向量AB和BC
    AB = (B[0] - A[0], B[1] - A[1])
    # BC = (C[0] - B[0], C[1] - B[1])
    BC = (B[0] - C[0], B[1] - C[1])

# 计算点积
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

# 计算模
    mod_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    mod_BC = math.sqrt(BC[0]**2 + BC[1]**2)

# 计算夹角的余弦值
    cos_theta = dot_product / (mod_AB * mod_BC)

# 计算夹角
    theta = math.acos(cos_theta)

    return theta
# 将夹角从弧度转换为度
    theta_degrees = math.degrees(theta)
    return theta_degrees

# 极坐标数据
#r = np.array([1, 2, 3, 4])
#theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
r = np.zeros(10)
theta = np.zeros(10)
ir = np.zeros(10)
it = np.zeros(10)

p = [0,100,98,112,105,98,112,105,98,112]
a = [0,0,40.10,80.21,119.75,159.86,199.86,240.07,280.17,320.28]
for i in range(10):
    r[i] = p[i]
    theta[i] = a[i]*np.pi/180
    if i == 0:
        ir[i] = 0
        it[i] = 0
    else:
        ir[i] = 100
        it[i] = 2*(i-1)*np.pi/9
    i += 1

# 将极坐标转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

ix = ir * np.cos(it)
iy = ir * np.sin(it)
"""
# 绘制图形
fig, ax = plt.subplots()
plt.plot(x, y, 'o', label = 'Original Position')
plt.plot(ix,iy, '*', label = 'Ideal Position')
circle = Circle((0, 0), 100, edgecolor='orange', facecolor='none')
ax.add_patch(circle)
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Ideal drown position')
plt.axis('equal')
plt.savefig('ques1-3')
"""
R = 100
eps = 1e-4
ansmean = []
ansmax = []
cnt = 2
while (True):
    cnt += 1
    # 计算欧几里得距离矩阵
    distances = np.sqrt((ix - x)**2 + (iy - y)**2)
    # 使用argmin找到最小距离的索引
    #print(distances)
    min_index = 2 + np.argmin(distances[2:])
    print(min_index)
    for i in range(2,10):
        if i == min_index:
            continue
        # alpha为弧度制
        alpha1 = calc_angle(x[0],y[0],x[i],y[i],x[1],y[1])
        alpha2 = calc_angle(x[1],y[1],x[i],y[i],x[min_index],y[min_index])
        alpha3 = calc_angle(x[0],y[0],x[i],y[i],x[min_index],y[min_index])
        # print(alpha1,alpha2,alpha3)
        # 转化为极坐标,就是θj
        thetaj = math.atan2(y[min_index],x[min_index])
        if alpha1+alpha2-alpha3<eps:
            if thetaj > 0:
                thetak = math.atan2(math.sin(alpha3)*math.sin(alpha1)
                                -math.sin(alpha1)*math.sin(alpha3-thetaj),
                                math.sin(alpha1)*math.cos(alpha3-thetaj)
                                -math.sin(alpha3)*math.cos(alpha1))
                rk=R*math.sin(alpha1+thetak)/math.sin(alpha1)
            else:
                thetak = math.atan2(math.sin(alpha1)*math.sin(alpha3+thetaj)
                                -math.sin(alpha3)*math.sin(alpha1),
                                math.sin(alpha1)*math.cos(alpha3+thetaj)
                                -math.sin(alpha3)*math.cos(alpha1))
                rk=R*math.sin(alpha1-thetak)/math.sin(alpha1)
        elif alpha1+alpha3-alpha2<eps:
            # atan2返回弧度制[-pi,pi]
            #if thetaj > 0:
            if alpha2 > 0.5*np.pi or thetaj > 0:
                thetak = math.atan2(math.sin(alpha3)*math.sin(alpha1)
                                -math.sin(alpha1)*math.sin(alpha3-thetaj),
                                math.sin(alpha1)*math.cos(alpha3-thetaj)
                                +math.sin(alpha3)*math.cos(alpha1))
                rk=R*math.sin(alpha1-thetak)/math.sin(alpha1)
                #print(21,i)
            else:
                thetak = math.atan2(math.sin(alpha1)*math.sin(alpha3+thetaj)
                                -math.sin(alpha3)*math.sin(alpha1),
                                math.sin(alpha1)*math.cos(alpha3+thetaj)
                                +math.sin(alpha3)*math.cos(alpha1))
                rk=R*math.sin(alpha1+thetak)/math.sin(alpha1)
                #print(22,i)
        
        elif alpha2+alpha3-alpha1<eps:
            if thetaj > 0:
                thetak = math.atan2(math.sin(alpha3)*math.sin(alpha1)
                                -math.sin(alpha1)*math.sin(alpha3+thetaj),
                                math.sin(alpha3)*math.cos(alpha1)-
                                math.sin(alpha1)*math.cos(alpha3+thetaj))
                rk=R*math.sin(alpha1-thetak)/math.sin(alpha1)
            else:
                thetak = math.atan2(math.sin(alpha1)*math.sin(alpha3-thetaj)
                                -math.sin(alpha3)*math.sin(alpha1),
                                math.sin(alpha3)*math.cos(alpha1)-
                                math.sin(alpha1)*math.cos(alpha3-thetaj))
                rk=R*math.sin(alpha1+thetak)/math.sin(alpha1)
        kx = rk*math.cos(thetak)
        ky = rk*math.sin(thetak)
        #if min_index == 7:
        #    print(x[i],y[i],kx,ky)
        x[i]=x[i]-kx+ix[i]
        y[i]=y[i]-ky+iy[i]

    fig, ax = plt.subplots()
    plt.plot(x, y, 'o', label = 'Actual Position')
    plt.plot(ix,iy, '*', label = 'Ideal Position')
    circle = Circle((0, 0), 100, edgecolor='orange', facecolor='none')
    ax.add_patch(circle)
    plt.xlabel('X_axis')
    plt.ylabel('Y_axis')
    plt.title('Actual and Standard drown position')
    plt.axis('equal')
    plt.savefig('ques13'+str(cnt))

    dis = np.sqrt((ix - x)**2 + (iy - y)**2)
    ansmean.append(np.mean(dis))
    ansmax.append(np.max(dis))
    print(dis)
    if np.max(dis) < 1e-2:
        break

df1 = pd.DataFrame(ansmean, columns=['mean'])
df2 = pd.DataFrame(ansmax, columns=['max'])
fig, ax = plt.subplots()
plt.title('Mean and Max distance between Actual position and Standard Position')
plt.plot(df1['mean'], label = 'mean distance')
plt.plot(df2['max'], label = 'max distance')
plt.legend()
plt.savefig('ques132')

print(ansmax)