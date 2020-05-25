''' keras40을 복사했음 '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras39_split import split_x

# 2. 모델구성
model = Sequential()
model.add(LSTM(10,activation='relu',input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(10))

model.summary()

model.save(".//model//Save_keras44.h5")
# model.save("./model/Save_44.h5")
# model.save(".\model\Save_44.h5")

print("저장이 잘 됐다.")