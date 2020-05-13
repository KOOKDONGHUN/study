# 1.데이터 생성 
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


# 2.모델구성
from keras.models import Sequential # 층을 구성하는 인풋에서 아웃풋으로 바로갈수 없으므로 중간을 거쳐 간다는 의미 계층구조
from keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim = 1,activation='relu'))


# 3. 훈련
model.compile(loss = 'mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=500,batch_size=1)


# 4. 평가 예측
los, acc = model.evaluate(x, y, batch_size =1)

print("loss : " ,los )
print("acc : " ,acc )