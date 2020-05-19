# keras14_mlp를 sequential에서 함수형으로 변경 multi layer perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from sklearn.metrics import r2_score
from keras.models import Model
import numpy as np

x = np.array([range(1,101),range(311,411),range(100)]).transpose()
y = np.array([range(101,201),range(711,811),range(100)]).transpose()

x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    train_size=0.95
    )

input1 = Input(shape=(3,))
dense1 = Dense(50)(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
output1 = Dense(3)(dense1)
model = Model(inputs=input1,outputs=output1)

els = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=100, batch_size=3,
            validation_split=0.3,callbacks=[els])
loss,mse = model.evaluate(x_test,y_test,batch_size=3)
y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

r2_y_predict = r2_score(y_test,y_predict)

print(y_test)
print(y_predict)
print("RMSE : ", RMSE(y_test,y_predict))
print("r2 : ",r2_y_predict)