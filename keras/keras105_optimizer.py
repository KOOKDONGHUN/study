import numpy as np

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

# 
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad

# Adam은 경사하강법을 이용한다
# optimize = Adam(learning_rate=0.001) # 0.07869881391525269
optimize = RMSprop(lr=0.001) # 0.03516753017902374
# optimize = SGD(learning_rate=0.001) # 0.06709844619035721
# optimize = Adadelta(learning_rate=0.001) # 5.948471546173096
# optimize = Adagrad(learning_rate=0.001) # 1.6297619342803955




model.compile(loss='mse',optimizer=optimize,metrics=['mse'])
model.fit(x,y,epochs=30,batch_size=2)

loss = model.evaluate(x,y)
print(loss)

pred1 = model.predict([3,5])
print(f'pred1 : {pred1}')