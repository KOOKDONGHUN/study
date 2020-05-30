from sklearn.datasets import load_boston
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
a_data = load_boston()
print(f"data : {a_data}")
print(f"data.type : {type(a_data)}")

x_data =a_data.data # 이거 빨간줄 뜨는거 데이터 타입이 사이킷런 bunch라는 건데 파이썬에서는 딕 문법이라서? 그런듯?
# print(f"x_train : {x_train}")
print(f"x_data.shape : {x_data.shape}") # (506, 13)

y_data =a_data.target
# print(f"y_train : {y_train}")
print(f"y_data.shape : {y_data.shape}") # (506,)

feature_names = a_data.feature_names
print(f"feature_names : {feature_names}") # 13개의 칼럼

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

std = StandardScaler()
std.fit(x_data) # (13,3)
x_data = std.transform(x_data)

pca = PCA(n_components=9)
pca.fit(x_data)

x_data = pca.fit_transform(x_data)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print('선택할 차원 수 :', d)

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=False,
    train_size=0.8
    )

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (404, 13)

'''
# 2. 모델
model = Sequential()
model.add(Dense(64,input_shape=(9,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

hist = model.fit(x_train,y_train,epochs=75,batch_size=6,callbacks=[],verbose=2,validation_split=0.1)

loss = hist.history['loss']
acc = hist.history['mse']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_mse']

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras74 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss') 
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras74 acc plot')

plt.plot(hist.history['val_mse'])
plt.plot(hist.history['mse'])

plt. grid()

plt.ylabel('mse')
plt.xlabel('epoch')

plt.legend(['train mse','val mse'])

plt.show()

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=6)

# R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score

y_predict = model.predict(x_test)

r2_y_predict = r2_score(y_test,y_predict)

print("r2 : ",r2_y_predict)
print(f"loss : {loss}")
# print(f"acc : {acc}") # acc : '''