import pandas as pd
import numpy as np
import csv
from hamsu import view_nan, split_x, plot_feature_importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv('./Data/Seoul/Seoul_data.csv')
print(data)
data = data.drop(['date'],axis=1)
print(data)

data = data.values

x_data = list()
y_data = list()

print(data.shape)
for i in range(len(data[:,0])-6):
    x_data.append(data[i:i+4,:])
    y_data.append(data[i+5,:3])

x_data = np.array(x_data)
y_data = np.array(y_data)

print(i)
x_train,x_test, y_train,y_test = train_test_split(x_data,y_data,shuffle=False,test_size=0.1)

print(x_train.shape)
print(x_train)
print(x_test)
model = RandomForestRegressor()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

for i in range(len(y_pred)):
    print(f"y_tets : {y_test[i]} \t y_pred : {y_pred[i].round(1)}")
        