import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
datas=pd.read_excel('Attachment2.xlsx')
X=datas[['系统I温度 (Temperature of system I)','系统II温度 (Temperature of system II)','原矿参数1 (Mineral parameter 1)','原矿参数2 (Mineral parameter 2)','原矿参数3 (Mineral parameter 3)','原矿参数4 (Mineral parameter 4)','原矿质量']]
Y_1=datas['指标A (index A)']
Y_2=datas['指标B (index B)']
Y_3=datas['指标C (index C)']
Y_4=datas['指标D (index D)']

X_1_Train,X_1_Test,Y_1_Train,Y_1_Test=train_test_split(X,Y_1,test_size=0.2,random_state=42)

RFR_1=RandomForestRegressor()
RFR_1.fit(X_1_Train,Y_1_Train)
Y_1_Pred=RFR_1.predict(X_1_Test)
Predqua1=pd.DataFrame({"实际值":Y_1_Test,"预测值":Y_1_Pred})
print(Predqua1)

X_2_Train,X_2_Test,Y_2_Train,Y_2_Test=train_test_split(X,Y_2,test_size=0.2,random_state=42)

RFR_2=RandomForestRegressor()
RFR_2.fit(X_2_Train,Y_2_Train)
Y_2_Pred=RFR_2.predict(X_2_Test)
Predqua2=pd.DataFrame({"实际值":Y_2_Test,"预测值":Y_2_Pred})
print(Predqua2)

X_3_Train,X_3_Test,Y_3_Train,Y_3_Test=train_test_split(X,Y_3,test_size=0.2,random_state=42)

RFR_3=RandomForestRegressor()
RFR_3.fit(X_3_Train,Y_3_Train)
Y_3_Pred=RFR_3.predict(X_3_Test)
Predqua3=pd.DataFrame({"实际值":Y_3_Test,"预测值":Y_3_Pred})
print(Predqua3)

X_4_Train,X_4_Test,Y_4_Train,Y_4_Test=train_test_split(X,Y_4,test_size=0.2,random_state=42)

RFR_4=RandomForestRegressor()
RFR_4.fit(X_4_Train,Y_4_Train)
Y_4_Pred=RFR_4.predict(X_4_Test)
Predqua4=pd.DataFrame({"实际值":Y_4_Test,"预测值":Y_4_Pred})
print(Predqua4)
Predqua_total=[RFR_1.predict(X),RFR_2.predict(X),RFR_3.predict(X),RFR_4.predict(X)]
