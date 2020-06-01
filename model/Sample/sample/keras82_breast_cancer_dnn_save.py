from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
a_data = load_breast_cancer()

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
y_data =a_data.target

feature_names = a_data.feature_names

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=True,
    train_size=0.8
    )

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 2. 모델
model = Sequential()
model.add(Dense(64,input_shape=(30,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(2))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
modelpath = './model/Sample/model/CANCER/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=3,callbacks=[chpoint],verbose=2,validation_split=0.1)

model.save('./model/Sample/model/CANCER/keras82_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/CANCER/keras82_weight.h5') # 가중치만 저장됨

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=3)

print(f"loss : {loss}") 
print(f"acc : {acc}") # acc : 