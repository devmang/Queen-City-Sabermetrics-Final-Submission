import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
'''
era_try = pd.read_csv('ERA_Training_Data.csv')

x = era_try.drop(columns= ['Delta ERA'] )
y = era_try['Delta ERA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)
c = lr.intercept_
m = lr.coef_

y_pred_train = lr.predict(x_train)

r2 = r2_score(y_train, y_pred_train)
print(r2)

y_pred_test = lr.predict(x_test)

#test_r2 = r2_score(y_test, y_pred_test)
#print(test_r2)

plt.scatter(y_train, y_pred_train)
plt.xlabel("actual delta ERA")
plt.ylabel("predicted delta ERA")
plt.show()

'''
rp_data = pd.read_csv('Data Reprocess/all_RP_w_TenPlusInnings_w_IdealColumns.csv')
rp_data = rp_data.dropna()



#pred_rp = lr.predict(rp_data)
#print(pred_rp)

# Export rp_data as a CSV file
rp_data.to_csv('all_RP_w_TenPlusInnings_w_IdealColumns_w_NoNaN.csv', index=False)
