import pandas as pd
import numpy as np
qua_data=pd.read_excel('qua_data.xlsx')
tem_data=pd.read_excel('tem_data.xlsx')
#print(qua_data,tem_data)
#t1_data = tem_data['系统I温度 (Temperature of system I)']
#t2_data = tem_data['系统II温度 (Temperature of system II)']
#time_data = tem_data['时间 (Time)']
index_1=qua_data['指标A (index A)']
index_2=qua_data['指标B (index B)']
index_3=qua_data['指标C (index C)']
index_4=qua_data['指标D (index D)']

#使用随机森林算法

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X=tem_data[['系统I温度 (Temperature of system I)','系统II温度 (Temperature of system II)']]
#indexA
Y_1=index_1

X_1_Train,X_1_Test,Y_1_Train,Y_1_Test=train_test_split(X,Y_1,test_size=0.2,random_state=42)

RFR_1=RandomForestRegressor()
RFR_1.fit(X_1_Train,Y_1_Train)
Y_1_Pred=RFR_1.predict(X_1_Test)
Predqua1=pd.DataFrame({"实际值":Y_1_Test,"预测值":Y_1_Pred})
#print(Predqua1)

#检验模型
RFRMeanSquaredError1=mean_squared_error(Y_1_Test,Y_1_Pred)
print("随机森林回归-MSE:%f"% RFRMeanSquaredError1)

#indexB
Y_2=index_2

X_2_Train,X_2_Test,Y_2_Train,Y_2_Test=train_test_split(X,Y_2,test_size=0.2,random_state=42)

RFR_2=RandomForestRegressor()
RFR_2.fit(X_2_Train,Y_2_Train)
Y_2_Pred=RFR_2.predict(X_2_Test)
Predqua2=pd.DataFrame({"实际值":Y_2_Test,"预测值":Y_2_Pred})
#print(Predqua2)

#检验模型
RFRMeanSquaredError2=mean_squared_error(Y_2_Test,Y_2_Pred)
print("随机森林回归-MSE:%f"% RFRMeanSquaredError2)

#indexC
Y_3=index_3

X_3_Train,X_3_Test,Y_3_Train,Y_3_Test=train_test_split(X,Y_3,test_size=0.2,random_state=42)

RFR_3=RandomForestRegressor()
RFR_3.fit(X_3_Train,Y_3_Train)
Y_3_Pred=RFR_3.predict(X_3_Test)
Predqua3=pd.DataFrame({"实际值":Y_3_Test,"预测值":Y_3_Pred})
#print(Predqua3)

#检验模型
RFRMeanSquaredError3=mean_squared_error(Y_3_Test,Y_3_Pred)
print("随机森林回归-MSE:%f"% RFRMeanSquaredError3)

#indexD
Y_4=index_4

X_4_Train,X_4_Test,Y_4_Train,Y_4_Test=train_test_split(X,Y_4,test_size=0.2,random_state=42)

RFR_4=RandomForestRegressor()
RFR_4.fit(X_4_Train,Y_4_Train)
Y_4_Pred=RFR_4.predict(X_4_Test)
Predqua4=pd.DataFrame({"实际值":Y_4_Test,"预测值":Y_4_Pred})
#print(Predqua4)

#检验模型
RFRMeanSquaredError4=mean_squared_error(Y_4_Test,Y_4_Pred)
print("随机森林回归-MSE:%f"% RFRMeanSquaredError4)

#tem_A=[1404.89,859.77]
#tem_B=[1151.75,859.77]
#tem_A_array = np.array([tem_A])
#tem_B_array = np.array([tem_B])
#print(RFR_1.predict(tem_A_array))
#print(RFR_2.predict(tem_A_array))
#print(RFR_3.predict(tem_A_array))
#print(RFR_4.predict(tem_A_array))
#print(RFR_1.predict(tem_B_array))
#print(RFR_2.predict(tem_B_array))
#print(RFR_3.predict(tem_B_array))
#print(RFR_4.predict(tem_B_array))
