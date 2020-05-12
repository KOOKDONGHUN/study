# 1.데이터 생성 
import numpy as np

ls = []
for i in range(50001):
    ls.append(i)


train_list = ls[0:49991]
test_list = ls[49990:50001]

# print(train_list)
# print(test_list)

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
model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_train, y_train))

# 4. 평가 예측
los, acc = model.evaluate(x_test, y_test, batch_size =100) #머신에게 훈련 데이터와 평가 데이터를 나눠서 학습과 평가를 하기 위함

print("loss : " ,los )
print("acc : " ,acc )

output = (model.predict(x_test))
print(output)