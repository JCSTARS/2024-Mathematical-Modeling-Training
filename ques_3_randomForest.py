import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
def is_qualified(x):
    return 77.78 < x[1] < 80.33 and x[2] < 24.15 and x[3] < 17.15 and x[4] < 15.62

datas=pd.read_excel('attachment2.xlsx')
datas=datas.head(1725)
if datas.isnull().values.any():
    # 删除含有 NaN 的行
    datas.dropna(inplace=True)
X=datas[['系统I温度 (Temperature of system I)','系统II温度 (Temperature of system II)','原矿参数1 (Mineral parameter 1)','原矿参数2 (Mineral parameter 2)','原矿参数3 (Mineral parameter 3)','原矿参数4 (Mineral parameter 4)','原矿质量']]
Y_1=datas['指标A (index A)']
Y_2=datas['指标B (index B)']
Y_3=datas['指标C (index C)']
Y_4=datas['指标D (index D)']

X_1_Train,X_1_Test,Y_1_Train,Y_1_Test=train_test_split(X,Y_1,test_size=0.2,random_state=42)

#print(X_1_Train, Y_1_Train)
RFR_1=RandomForestRegressor()
RFR_1.fit(X_1_Train,Y_1_Train)
Y_1_Pred=RFR_1.predict(X_1_Test)
Predqua1=pd.DataFrame({"实际值":Y_1_Test,"预测值":Y_1_Pred})
#print(Predqua1)

X_2_Train,X_2_Test,Y_2_Train,Y_2_Test=train_test_split(X,Y_2,test_size=0.2,random_state=42)

RFR_2=RandomForestRegressor()
RFR_2.fit(X_2_Train,Y_2_Train)
Y_2_Pred=RFR_2.predict(X_2_Test)
Predqua2=pd.DataFrame({"实际值":Y_2_Test,"预测值":Y_2_Pred})
#print(Predqua2)

X_3_Train,X_3_Test,Y_3_Train,Y_3_Test=train_test_split(X,Y_3,test_size=0.2,random_state=42)

RFR_3=RandomForestRegressor()
RFR_3.fit(X_3_Train,Y_3_Train)
Y_3_Pred=RFR_3.predict(X_3_Test)
Predqua3=pd.DataFrame({"实际值":Y_3_Test,"预测值":Y_3_Pred})
#print(Predqua3)

X_4_Train,X_4_Test,Y_4_Train,Y_4_Test=train_test_split(X,Y_4,test_size=0.2,random_state=42)

RFR_4=RandomForestRegressor()
RFR_4.fit(X_4_Train,Y_4_Train)
Y_4_Pred=RFR_4.predict(X_4_Test)
Predqua4=pd.DataFrame({"实际值":Y_4_Test,"预测值":Y_4_Pred})
#print(Predqua4)

# 使用每个模型对整个特征集 X 进行预测
Y_1_Pred = RFR_1.predict(X)
Y_2_Pred = RFR_2.predict(X)
Y_3_Pred = RFR_3.predict(X)
Y_4_Pred = RFR_4.predict(X)

# 将预测结果转换为DataFrame
Predqua1 = pd.DataFrame(Y_1_Pred, columns=['指标A (index A)_预测值'])
Predqua2 = pd.DataFrame(Y_2_Pred, columns=['指标B (index B)_预测值'])
Predqua3 = pd.DataFrame(Y_3_Pred, columns=['指标C (index C)_预测值'])
Predqua4 = pd.DataFrame(Y_4_Pred, columns=['指标D (index D)_预测值'])

# 使用 pd.concat 沿着列方向合并预测结果
Predqua_total = pd.concat([Predqua1, Predqua2, Predqua3, Predqua4], axis=1)

# 打印结果
#print(Predqua_total)

# 修改 is_qualified 函数以适应单列输入
def is_qualified_single_column(preds, index_A_col, index_B_col, index_C_col, index_D_col):
    return (77.78 < preds[index_A_col] < 80.33) and (preds[index_B_col] < 24.15) and (preds[index_C_col] < 17.15) and (preds[index_D_col] < 15.62)

# 应用函数到 Predqua_total 的每一行
Predqua_total['合格'] = Predqua_total.apply(is_qualified_single_column,
                                            args=(['指标A (index A)_预测值', '指标B (index B)_预测值', '指标C (index C)_预测值', '指标D (index D)_预测值']),
                                            axis=1)

#print(Predqua_total['合格'])
#Predqua_total['合格'].to_excel("haha.xlsx")
#print('AUC:', roc_auc_score(datas['是否合格'],Predqua_total['合格'] ))

testA1=[341.40,665.04,52.88,91.27,47.22,22.26,470.31]
testA2=[341.40,665.04,52.88,91.27,47.22,22.26,479.71]
testA3=[341.40,665.04,52.88,91.27,47.22,22.26,470.04]
testA4=[341.40,665.04,52.88,91.27,47.22,22.26,444.46]
testA5=[341.40,665.04,52.88,91.27,47.22,22.26,444.67]
testA6=[341.40,665.04,52.88,91.27,47.22,22.26,449.67]
testA7=[341.40,665.04,52.88,91.27,47.22,22.26,457.1]
testA8=[341.40,665.04,52.88,91.27,47.22,22.26,437.68]
testB1=[1010.32,874.47,54.44,92.12,48.85,21.83,453.56]
testB2=[1010.32,874.47,54.44,92.12,48.85,21.83,450.9]
testB3=[1010.32,874.47,54.44,92.12,48.85,21.83,437.04]
testB4=[1010.32,874.47,54.44,92.12,48.85,21.83,426.66]
testB5=[1010.32,874.47,54.44,92.12,48.85,21.83,421.47]
testB6=[1010.32,874.47,54.44,92.12,48.85,21.83,433.41]
testB7=[1010.32,874.47,54.44,92.12,48.85,21.83,403.17]
testB8=[1010.32,874.47,54.44,92.12,48.85,21.83,450.13]

testA1_array = np.array(testA1)
testA2_array = np.array(testA2)
testA3_array = np.array(testA3)
testA4_array = np.array(testA4)
testA5_array = np.array(testA5)
testA6_array = np.array(testA6)
testA7_array = np.array(testA7)
testA8_array = np.array(testA8)
testB1_array = np.array(testB1)
testB2_array = np.array(testB2)
testB3_array = np.array(testB3)
testB4_array = np.array(testB4)
testB5_array = np.array(testB5)
testB6_array = np.array(testB6)
testB7_array = np.array(testB7)
testB8_array = np.array(testB8)

TA1A_Pred = RFR_1.predict([testB8_array])
TA1B_Pred = RFR_2.predict([testB8_array])
TA1C_Pred = RFR_3.predict([testB8_array])
TA1D_Pred = RFR_4.predict([testB8_array])
print(TA1A_Pred)
print(TA1B_Pred)
print(TA1C_Pred)
print(TA1D_Pred)
