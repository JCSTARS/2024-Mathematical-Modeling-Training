import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
# 通过sheet名称读取
def get_sheet():
    sheet1 = pd.read_excel("附件1(Attachment 1)2022-51MCM-Problem B.xlsx", sheet_name=0)
    sheet2 = pd.read_excel("附件1(Attachment 1)2022-51MCM-Problem B.xlsx", sheet_name=1)
    sheet3 = pd.read_excel("附件1(Attachment 1)2022-51MCM-Problem B.xlsx", sheet_name=2)
    return sheet1, sheet2, sheet3

def tem_qua_data_pre(sheet1, sheet2):
    """
    系统温度数据预处理
    从图中不难看出温度是阶段性变化的
    并且根据后续产品质量与时间关系, 只有在之前(?:50)2小时的数据有意义, 二者一一对应
    同时存在一些数据是需要舍去的
    """
    # 构建温度曲线 方便观察
    t1_data = sheet1['系统I温度 (Temperature of system I)']
    t2_data = sheet1['系统II温度 (Temperature of system II)']
    time_data = sheet1['时间 (Time)']
    plt.figure(figsize = (18, 8))
    plt.plot(time_data, t1_data, label = 'System(I) temperature')
    plt.plot(time_data, t2_data, label = 'System(II) temperature')
    # 设置日期格式器
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # 设置日期刻度定位器
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # 添加图例
    plt.legend(loc = 'lower right')
    # 可选：设置图表标题和轴标签
    plt.title('System temperature(I, II) and time')
    plt.savefig('temperature-time.jpg')
    """
    进行进一步温度数据处理
    先找到分钟是50的数据
    """
    cond1= sheet1.iloc[:, 0].astype('string').apply(lambda x: x[14: 16]) == "50"
    tem_data = sheet1[cond1].iloc[:-2, :]
    tem_data.index = [i for i in range(len(tem_data))]

    """
    处理有效的矿石质量对应温度数据
    最前面的两个数据不能用，因为矿石投进去的时候我们尚未记录温度
    注意到1-20 6:25-9：04之间的数据是缺乏的，因此矿石质量数据也不能使用
    """
    quality_time= sheet2.iloc[:, 0].astype('string').apply(lambda x: x)
    # 舍弃3个数据
    exp_date = ["2022-01-20 08:50:00", "2022-01-20 09:50:00", "2022-01-20 10:50:00"]
    cond2 = quality_time.apply(lambda x: x == exp_date[0] or x == exp_date[1] or x == exp_date[2])
    qua_data = sheet2[cond2.apply(lambda x: not x)].iloc[2:, :]
    qua_data.index = [i for i in range(len(qua_data))]
    # 此时qua_data与tem_data都是0~234的序列， 并且一一对应
    return tem_data, qua_data

sh1, sh2, sh3 = get_sheet()
print(tem_qua_data_pre(sh1, sh2))
