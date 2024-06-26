import pandas as pd

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

Y_1=index_1

X_Train,X_Test,Y_1_Train,Y_1_Test=train_test_split(X,Y_1,test_size=0.2,random_state=42)

RFR_1=RandomForestRegressor()
RFR_1.fit(X_Train,Y_1_Train)
Y_1_Pred=RFR_1.predict(X_Test)
Predqua=pd.DataFrame({"实际值":Y_1_Test,"预测值":Y_1_Pred})
print(Predqua)

#检验模型
RFRMeanSquaredError=mean_squared_error(Y_1_Test,Y_1_Pred)
print("随机森林回归-MSE:%f"% RFRMeanSquaredError)