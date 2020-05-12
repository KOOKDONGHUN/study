import numpy as np
# 데이터 생성 
train_list = []
for i in range(2000):
    train_list.append(i)

x_train = np.array(train_list)
y_train = np.array(train_list)

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim = 1,activation='relu'))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1,activation='relu'))

model.summary()

model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=100, validation_data=(x_train, y_train))
los, acc = model.evaluate(x_test, y_test, batch_size =100)

print("loss : " ,los )
print("acc : " ,acc )

output = (model.predict(x_test))
print(output)