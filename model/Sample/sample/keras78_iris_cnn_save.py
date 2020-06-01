from sklearn.datasets import load_iris
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
a_data = load_iris()

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
y_data =a_data.target

feature_names = a_data.feature_names

# std = StandardScaler()
# std.fit(x_data) # (150,4)
# x_data = std.transform(x_data)

# pca = PCA(n_components=9)
# pca = PCA()
# pca.fit(x_data)

# x_data = pca.fit_transform(x_data)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print('선택할 차원 수 :', d)

x_data = x_data.reshape(x_data.shape[0],2,2,1)
y_data = np_utils.to_categorical(y_data)

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=True,
    train_size=0.8
    )

# 2.  모델
model = Sequential()
model.add(Conv2D(50,(2,2),input_shape=(2,2,1),activation='relu'))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
modelpath = './model/Sample/model/IRIS/{epoch:02d}--{val_loss:.4f}.hdf5'
chpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                          save_best_only=True,save_weights_only=False,mode='auto',verbose=1)

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=3,callbacks=[chpoint],verbose=2,validation_split=0.03)

model.save('./model/Sample/model/IRIS/keras78_model.h5') # 가중치 까지 저장됨
model.save_weights('./model/Sample/model/IRIS/keras78_weight.h5') # 가중치만 저장됨

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=3)

print(f"loss : {loss}") 
print(f"acc : {acc}") # acc : 