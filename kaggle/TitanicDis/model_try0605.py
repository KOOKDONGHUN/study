import pandas as pd # 판다스데이터 프레임으로 구조를 만들때는 인자로 딕의 형태를 받음 -> pd.DataFrame(dic)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
x_data = np.load('./data/npy/titanic_x.npy')
y_data = np.load('./data/npy/titanic_y.npy')
test_data = np.load('./data/npy/titanic_test.npy')

y_data = np_utils.to_categorical(y_data)

std = StandardScaler()
x_data = std.fit_transform(x_data)
test_data = std.transform(test_data)


# 총 891행의 데이터가 각각 14개의 칼럼을 가지고 있음
# ---------------딥러닝 시작 ----------------

# 모델 구성
model = Sequential()
model.add(Dense(180,input_dim=14,activation='relu'))
model.add(Dropout(0.55))
model.add(Dense(150))
model.add(Dropout(0.68))
model.add(Dense(118))
model.add(Dropout(0.58))
model.add(Dense(150))
model.add(Dropout(0.68))
model.add(Dense(150))
model.add(Dropout(0.75))

model.add(Dense(50))

model.add(Dense(2,activation='sigmoid'))

model.summary()

# 컴파일 및 실행

print(x_data[:,1:])
print(x_data[:,1:].shape)
print(y_data)
print(y_data.shape)

els = EarlyStopping(monitor='loss', patience=8, mode='auto')

model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_data[:,1:],y_data,epochs=200,batch_size=32,callbacks=[els],verbose=2,validation_split=0.03)


pred = model.predict(test_data[:,1:])
pred = np.argmax(pred,axis=1).astype(int)

print("pred : \n",pred)
print("test_data : \n",test_data[:,0])

test_data = std.inverse_transform(test_data)

submission = pd.DataFrame({
    "PassengerId": test_data[:,0].astype(int),
    "Survived": pred
})

submission.to_csv('./submit/submission_dnn.csv', index = False)

'''validation을 0.1에서 0.05로 줄이고, 전체 drop의 비율을 0.7에서 살짝 낮췄을때 0.02의 상승이 있었음
   validation이 0.1 일때 레이어의 갯수를 늘려보았지만 오히려 0.1이상의 하락이 있었음
   '''