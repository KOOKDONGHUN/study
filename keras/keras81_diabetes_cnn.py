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


# 1. 데이터
a_data = load_iris()
print(f"data : {a_data}")
print(f"data.type : {type(a_data)}")

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
print(f"x_data : {x_data}")
print(f"x_data.shape : {x_data.shape}") # 150,4

y_data =a_data.target
print(f"y_data : {y_data}")
print(f"y_data.shape : {y_data.shape}") # 150,

feature_names = a_data.feature_names
print(f"feature_names : {feature_names}") # 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

print(f"x_data.shape : {x_data.shape}") # x_train.shape : (120, 4)

x_data = x_data.reshape(x_data.shape[0],2,2,1)
y_data = np_utils.to_categorical(y_data)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=True,
    train_size=0.8
    )

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (120, 2,2,1)

print(f"y_train : {y_train}")

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
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=30,batch_size=3,callbacks=[],verbose=2,validation_split=0.03)

# 4. 평가, 예측

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras78 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss') 
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras78 acc plot')

plt.plot(hist.history['val_acc'])
plt.plot(hist.history['acc'])

plt. grid()

plt.ylabel('acc')
plt.xlabel('epoch')

plt.legend(['train acc','val acc'])

plt.show()

loss,acc = model.evaluate(x_test,y_test,batch_size=3)
# print("r2 : ",r2_y_predict)
print(f"loss : {loss}") 
print(f"acc : {acc}") # acc : 