import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

era_try = pd.read_csv('WHIP_Training_Data.csv')

x = era_try.drop(columns= ['Delta WHIP'] )
y = era_try['Delta WHIP']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)
c = lr.intercept_
m = lr.coef_

y_pred_train = lr.predict(x_train)

r2 = r2_score(y_train, y_pred_train)
print(r2)

y_pred_test = lr.predict(x_test)


rp_data = pd.read_csv('Data Reprocess/all_RP_w_TenPlusInnings_w_IdealColumns_w_NoNaN.csv')
rp_data_adjusted = rp_data.drop(columns=['MLBAMID', 'WHIP'])
pred_rp = lr.predict(rp_data_adjusted)
print(pred_rp)

pred_rp_df = pd.DataFrame(pred_rp, columns=['Predicted Delta WHIP'])
pred_rp_df.to_csv('predicted_delta_WHIP123.csv', index=False)
