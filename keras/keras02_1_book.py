# 1.데이터 생성 
import numpy as np

train_list = [i for i in range(2000)]
test_list = [i for i in range(2000,4000)]

x_train = np.array(train_list)
y_train = np.array(train_list)

x_test = np.array(test_list)
y_test = np.array(test_list)

# 2.모델구성
from keras.models import Sequential # 층을 구성하는 인풋아서 아웃풋으로 바로갈수 없으므로 중간을 거쳐 간다는 의미
from keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim = 1,activation='relu'))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1,activation='relu'))

model.summary()

# 3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])  #loss(오차)를 mse로 계산해서 줄여나간다, optimizer->최적화, metrics->훈련과정에서 프린트 되는 부분을 accuracy로 하겠다.
model.fit(x_train, y_train, epochs=50, batch_size=20, validation_data=(x_train, y_train),callbacks=[els]) # 운동은 피트니스가서함-> 머신이 학습하는 장소
                                                                                          # validation_data=(x_train, y_train)는 한번의 epochs이 돌때 마다
                                                                                          # x_train, y_train 값을 가지고 테스트한 후 가중치를 수정하겠다.

# 4. 평가 예측
los, acc = model.evaluate(x_test, y_test, batch_size =20) #평가 데이터(x_test, y_test)를 가지고 평가를 하기 위함

print("loss : " ,los )
print("acc : " ,acc )

output = (model.predict(x_test))
print(output)
