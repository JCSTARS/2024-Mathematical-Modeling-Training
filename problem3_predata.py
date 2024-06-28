import pandas as pd
import numpy as np

file_path = './附件2(Attachment 2)2022-51MCM-Problem B.xlsx'
sheet1 = pd.read_excel(
    io=file_path, 
    index_col=None, 
    sheet_name='温度(temperature)', )
sheet2 = pd.read_excel(
    io=file_path, 
    index_col=None, 
    sheet_name='产品质量(quality of the products)', )
sheet3 = pd.read_excel(
    io=file_path, 
    index_col=None, 
    sheet_name='原矿参数(mineral parameter)', )
sheet4 = pd.read_excel(
    io=file_path, 
    index_col=None, 
    sheet_name='过程数据(process parameter)', )

# 找到有效的温度数据
sheet1_time_string = sheet1.iloc[:, 0].astype('string')

cond1 = sheet1_time_string.apply(lambda x: x[14: 16]) == "50"
data_part1 = sheet1[cond1].iloc[:-2, :]

exp_date1 = [
    "2022-02-03 20:50:00", 
    "2022-02-26 13:50:00", 
    "2022-03-21 06:50:00", 
    "2022-04-04 10:50:00", "2022-04-04 15:50:00", 
    "2022-03-10 10:50:00", "2022-03-10 11:50:00", "2022-03-10 12:50:00", 
]
cond1 = sheet1_time_string.apply(lambda x: x in exp_date1)
data_part1 = data_part1[cond1.apply(lambda x: not x)]
data_part1.index = [i for i in range(len(data_part1))]
# print(data_part1.shape)

# 找到有效的产品质量数据
sheet2_time_string = sheet2.iloc[:, 0].astype('string')
exp_date2 = [
    "2022-02-20 23:50:00", "2022-02-21 00:50:00", "2022-02-21 01:50:00", 
    "2022-02-21 09:50:00", "2022-02-21 10:50:00", "2022-02-21 02:50:00", "2022-02-21 03:50:00", "2022-02-21 04:50:00", 
    "2022-02-21 05:50:00", "2022-02-21 06:50:00", "2022-02-21 07:50:00", "2022-02-21 08:50:00", 
    "2022-02-26 06:50:00", "2022-02-26 07:50:00", "2022-02-26 08:50:00", "2022-02-26 09:50:00", "2022-02-26 10:50:00",
    "2022-04-08 00:50:00", "2022-04-08 01:50:00", 
]

cond2 = sheet2_time_string.apply(lambda x: x in exp_date2)
data_part2_ = sheet2[cond2.apply(lambda x: not x)].iloc[2:, :]
data_part2_.index = [i for i in range(len(data_part2_))]
# print(data_part2_.shape)

# todo 计算合格数量、合格率
cond10 = 77.78 < data_part2_.iloc[:, 1]
cond11 = data_part2_.iloc[:, 1] < 80.33
cond2 = data_part2_.iloc[:, 2] < 24.15
cond3 = data_part2_.iloc[:, 3] < 17.15
cond4 = data_part2_.iloc[:, 4] < 15.62
print("合格率：", len(data_part2_[cond10][cond11][cond2][cond3][cond4]) / len(data_part2_))

# todo 找到产品是否合格
def is_qualified(x):
    return 77.78 < x[1] < 80.33 and x[2] < 24.15 and x[3] < 17.15 and x[4] < 15.62
data_part2 = pd.DataFrame(data_part2_.apply(is_qualified, axis=1))
data_part2.columns = ['是否合格']

# print(len(data_part2))
# print(data_part2.sum() / len(data_part2))

# print(data_part2.shape)
# print(data_part2.sum(), data_part2.sum() / len(data_part2))

# todo 找到原矿参数数据 3
cnt = data_part1.iloc[:, 0].astype('string').apply(lambda x: x[5: 10])
time_cnt = []
for i in pd.DataFrame(cnt).groupby(by='时间 (Time)'):
    time_cnt.append(len(i[1]))
data_part3 = pd.DataFrame(np.repeat(sheet3.iloc[:-4, :].values, time_cnt, axis=0), columns=sheet3.columns)
print(data_part3.shape)
# data_part3
# mitosheet.sheet(data_part3, analysis_to_replay="id-rvruqimmlr")

cols = ['时间 (Time)', "过程数据3 (Process parameter 3)", "过程数据4 (Process parameter 4)"]
proc_data = pd.DataFrame(sheet4)
#proc_data.iplot(x='时间 (Time)')
print("相关系数：", proc_data.iloc[:, 1:].corr())


def norm(data):
    return (data - data.min()) / (data.max() - data.min())

data_part4 = sheet4.apply(lambda x: (x[3] + x[4]), axis=1)
data_part4 = pd.concat([sheet4.iloc[:, 0], data_part4], axis=1).rename(columns={0: "原矿质量"})
# print(data_part4.shape)
# data_part4
# mitosheet.sheet(data_part4, analysis_to_replay="id-lntnexsmmk")

exp_date4 = []
for i in exp_date1 + exp_date2:
    exp_date4.append(i[:-5] + "30")
# print(exp_date4)

sheet4_time_string = data_part4.iloc[:, 0].astype('string')
cond4 = sheet4_time_string.apply(lambda x: x[: -3] not in exp_date4)

data_part4_need = data_part4[cond4]
data_part4_need = data_part4_need.iloc[:-33, :]

for _ in range(5):
    data_part4_need.drop(index=np.random.randint(0, len(data_part4_need)), inplace=True)
data_part4_need.index = [i for i in range(len(data_part4_need))]

data_part4_need = pd.DataFrame(np.repeat(data_part4_need.values, 3, axis=0), columns=data_part4_need.columns)

proc_data = proc_data.iloc[:,3:5]
data_proc_need = pd.DataFrame(np.repeat(proc_data.values, 3, axis=0), columns=proc_data.columns)
# print(data_part4_need.shape)
# data_part4_need
# mitosheet.sheet(data_part4_need, analysis_to_replay="id-owygulbcev")

combined_data = pd.concat([data_part1, data_part2_, data_part2, data_part3, data_part4_need, data_proc_need], axis=1)
combined_data.to_excel('attachment2.xlsx', sheet_name='Sheet1', index=False)

X = pd.concat([data_part1.iloc[:, 1:], data_part3.iloc[:, 1:], data_part4_need.iloc[:, 1:]], axis=1)
Ys = data_part2

X.to_csv("quention3-X_data.csv")
Ys.to_csv("quention3-Y_data.csv")

cond = (pd.notna(X).iloc[:, 0] == True)
remain_index = X[cond].index

X = X[cond]
Y = Ys[cond].replace(to_replace=[True, False], value=[1, 0])
# print(X.shape, Y.shape)
 
datas =  data_part2_
# 假设时间列是第一列
datas['日期 (Date)'] = datas.iloc[:, 0].dt.date

# 应用is_qualified函数到DataFrame的行，并创建一个新列'是否合格'
datas['是否合格'] = datas.apply(is_qualified, axis = 1)

# 按日期分组并计算每天的合格率
daily_qualified_counts = datas.groupby('日期 (Date)')['是否合格'].sum()
daily_total_counts = datas.groupby('日期 (Date)')['是否合格'].count()
daily_pass_rate = (daily_qualified_counts / daily_total_counts) * 100

# 创建一个包含日期和合格率的DataFrame
daily_pass_rate_df = pd.DataFrame({'日期': daily_pass_rate.index, '合格率': daily_pass_rate})

datas =  data_part1
# 假设时间列是第一列
datas['日期 (Date)'] = datas.iloc[:, 0].dt.date
# print(datas)
# 按日期分组并计算每天的合格率
daily_tema1 = datas.groupby('日期 (Date)')['系统I温度 (Temperature of system I)'].mean()
daily_tema2 = datas.groupby('日期 (Date)')['系统II温度 (Temperature of system II)'].mean()
daily_temv1 = datas.groupby('日期 (Date)')['系统I温度 (Temperature of system I)'].var()
daily_temv2 = datas.groupby('日期 (Date)')['系统II温度 (Temperature of system II)'].var()
# 创建一个包含日期和合格率的DataFrame
daily_temp = pd.DataFrame({'温度1均值': daily_tema1,'温度2均值': daily_tema2,
                           '温度1方差': daily_temv1,'温度2方差': daily_temv2})
# print(daily_temp)

datas =  data_part3
# 假设时间列是第一列
datas['日期 (Date)'] = datas.iloc[:, 0].dt.date

daily_mine = pd.DataFrame({'原矿参数1': datas.groupby('日期 (Date)')['原矿参数1 (Mineral parameter 1)'].mean(),
                           '原矿参数2': datas.groupby('日期 (Date)')['原矿参数2 (Mineral parameter 2)'].mean(),
                           '原矿参数3': datas.groupby('日期 (Date)')['原矿参数3 (Mineral parameter 3)'].mean(),
                           '原矿参数4': datas.groupby('日期 (Date)')['原矿参数4 (Mineral parameter 4)'].mean()})

print(daily_mine)

datas = data_part4_need
# 假设时间列是第一列
datas['日期 (Date)'] = datas.iloc[:, 0].dt.date

# print(datas)
# 按日期分组并计算每天的合格率
daily_quaa = datas.groupby('日期 (Date)')['原矿质量'].mean()
daily_quav = datas.groupby('日期 (Date)')['原矿质量'].var()
# 创建一个包含日期和合格率的DataFrame
daily_qua = pd.DataFrame({'原矿质量均值': daily_quaa,'原矿质量方差': daily_quav})
print(daily_qua)

combined_data = pd.concat([daily_pass_rate_df, daily_temp, daily_mine, daily_qua], axis=1)
combined_data.dropna(how='any', inplace=True)
combined_data.to_excel('question4.xlsx', sheet_name='Sheet1', index=False)
