import numpy as np

x_train = np.array([i for i in range(1,11)])
y1_train = np.array([i for i in range(1,11)])


# model
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# compile, fit
model.compile(loss ='binary_crossentropy', optimizer='adam',metrics=['mse','acc'])

model.fit(x_train, y1_train, epochs=10, batch_size=2, verbose=2)

# predict
loss = model.evaluate(x_train,y1_train)
print(f'loss : {loss}')

x_pred = np.array([11,12,13,14])

y_pred = model.predict(x_pred)
print(f'y_pred : {y_pred}')