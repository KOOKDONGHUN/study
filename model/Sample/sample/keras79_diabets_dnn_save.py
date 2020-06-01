from sklearn.datasets import load_diabetes
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
a_data = load_diabetes()

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
y_data =a_data.target

feature_names = a_data.feature_names

std = StandardScaler()
std.fit(x_data) # (,)
x_data = std.transform(x_data)

# pca = PCA(n_components=9)
# pca.fit(x_data)

# x_data = pca.fit_transform(x_data)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print('선택할 차원 수 :', d)

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=False,
    train_size=0.8
    )

# 2. 모델
model = Sequential()
model.add(Dense(64,input_shape=(10,)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
modelpath = './model/Sample/model/DIABETS/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

hist = model.fit(x_train,y_train,epochs=25,batch_size=3,callbacks=[chpoint],verbose=2,validation_split=0.03)

model.save('./model/Sample/model/DIABETS/keras79_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/DIABETS/keras79_weight.h5') # 가중치만 저장됨

# 4. 평가, 예측
y_predict = model.predict(x_test)
r2_y_predict = r2_score(y_test,y_predict)

loss,mse = model.evaluate(x_test,y_test,batch_size=3)

print(f"mse : {mse}") 
print(f"loss : {loss}")
print("r2 : ",r2_y_predict)