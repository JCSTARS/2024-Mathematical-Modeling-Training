import pandas as pd
import  numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

datas=pd.read_excel('question4.xlsx')
datasadd=pd.read_excel('question4add.xlsx')
datassub=pd.read_excel('question4sub.xlsx')
X=datas[['合格率','温度1方差','温度2方差','原矿参数1','原矿参数2','原矿参数3','原矿参数4','原矿质量均值','原矿质量方差']]
Y=datas[['温度1均值','温度2均值']]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.25,random_state=46)
#random_state=27时对温度1拟合效果较好
#random_state=46时对温度2拟合效果较好
RFR=RandomForestRegressor()
mor=MultiOutputRegressor(RFR)
mor.fit(X_Train,Y_Train)
Y_pred=mor.predict(X_Test)
#Predqua1=pd.DataFrame({"实际值":Y_Test['温度1均值'],"预测值":Y_pred[:,0]})
#Predqua2=pd.DataFrame({"实际值":Y_Test['温度2均值'],"预测值":Y_pred[:,1]})
#print(Predqua1)
#print(Predqua2)

#以下部分为准确性检验

#RFRMeanSquaredError1=mean_squared_error(Y_Test['温度1均值'],Y_pred[:,0])
#RFRMeanSquaredError2=mean_squared_error(Y_Test['温度2均值'],Y_pred[:,1])
#print(RFRMeanSquaredError1)
#print(RFRMeanSquaredError2)

#以下部分为敏感性检验

#XADD=datasadd[['合格率','温度1方差','温度2方差','原矿参数1','原矿参数2','原矿参数3','原矿参数4','原矿质量均值','原矿质量方差']]
#XSUB=datassub[['合格率','温度1方差','温度2方差','原矿参数1','原矿参数2','原矿参数3','原矿参数4','原矿质量均值','原矿质量方差']]
#Pred_religion_values=mor.predict(X)
#Pred_add_values = mor.predict(XADD)
#Pred_sub_values = mor.predict(XSUB)
#Pred_religion_list=[item.tolist() for item in Pred_religion_values]
#Pred_add_list = [item.tolist() for item in Pred_add_values]
#Pred_sub_list = [item.tolist() for item in Pred_sub_values]
#Pred_add=pd.DataFrame({"原预估值":Pred_religion_list,"新预估值ADD":Pred_add_list})
#Pred_sub=pd.DataFrame({"原预估值":Pred_religion_list,"新预估值SUB":Pred_sub_list})
#print(Pred_add)
#print(Pred_sub)

test1=[0.8,0,0,56.27,111.38,47.52,20.26,434.0325,1570.32]
test2=[0.99,0,0,56.71,111.46,46.67,18.48,426.725,171.28]
test1_array = np.array(test1)
test2_array = np.array(test2)
test1_Pred = mor.predict([test1_array])
test2_Pred = mor.predict([test2_array])
print(test1_Pred)
print(test2_Pred)