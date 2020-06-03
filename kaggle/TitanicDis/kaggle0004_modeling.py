import pandas as pd # 판다스데이터 프레임으로 구조를 만들때는 인자로 딕의 형태를 받음 -> pd.DataFrame(dic)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

# 1. 데이터 불러오기
x_data = np.load('./data/titanic_x.npy')
y_data = np.load('./data/titanic_y.npy')
test_data = np.load('./data/titanic_test.npy')

y_data = np_utils.to_categorical(y_data)


# 총 891행의 데이터가 각각 14개의 칼럼을 가지고 있음
# ---------------딥러닝 시작 ----------------

# 모델 구성
model = Sequential()
model.add(Dense(200,input_dim=14,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(150))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))

model.add(Dense(2,activation='sigmoid'))

# 컴파일 및 실행

print(x_data[:,1:])
print(x_data[:,1:].shape)
print(y_data)
print(y_data.shape)

els = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_data[:,1:],y_data,epochs=200,batch_size=32,callbacks=[els],verbose=2)

pred = model.predict(test_data[:,1:])
pred = np.argmax(pred,axis=1).astype(int)

print("pred : \n",pred)
print("test_data : \n",test_data[:,0])

submission = pd.DataFrame({
    "PassengerId": test_data[:,0].astype(int),
    "Survived": pred
})

submission.to_csv('./submit/submission_dnn.csv', index = False)