import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
datas=pd.read_excel('附件1_总数据.xlsx')
X=datas[['指标A (index A)','指标B (index B)','指标C (index C)','指标D (index D)','原矿参数1 (Mineral parameter 1)','原矿参数2 (Mineral parameter 2)','原矿参数3 (Mineral parameter 3)','原矿参数4 (Mineral parameter 4)']]
Y=datas[['系统I温度 (Temperature of system I)','系统II温度 (Temperature of system II)']]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.25,random_state=45)
RFR=RandomForestRegressor()
mor=MultiOutputRegressor(RFR)
mor.fit(X_Train,Y_Train)
Y_pred=mor.predict(X_Test)
Predqua1=pd.DataFrame({"实际值":Y_Test['系统I温度 (Temperature of system I)'],"预测值":Y_pred[:,0]})
Predqua2=pd.DataFrame({"实际值":Y_Test['系统II温度 (Temperature of system II)'],"预测值":Y_pred[:,1]})
#print(Predqua1)
#print(Predqua2)
RFRMeanSquaredError1=mean_squared_error(Y_Test['系统I温度 (Temperature of system I)'],Y_pred[:,0])
RFRMeanSquaredError2=mean_squared_error(Y_Test['系统II温度 (Temperature of system II)'],Y_pred[:,1])
#print(RFRMeanSquaredError1)
#print(RFRMeanSquaredError2)
