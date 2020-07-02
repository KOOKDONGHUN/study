import numpy as np

x_train = np.array([i for i in range(1,11)])
y1_train = np.array([i for i in range(1,11)])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])


# model
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

in1 = Input(shape=(1,))
d1 = Dense(100)(in1)
d1 = Dense(100)(d1)
d1 = Dense(100)(d1)

in2 = Input(shape=(1,))
d2 = Dense(100)(in2)
d2 = Dense(100)(d2)
d2 = Dense(100)(d2)

mg = concatenate([d1,d2])
mg = Dense(100)(mg)

d1_1 = Dense(50)(mg)
out1 = Dense(1)(d1_1)

d2_2 = Dense(50)(mg)
out2 = Dense(1,activation='sigmoid')(d2_2)

model = Model(inputs=[in1,in2], outputs=[out1, out2])

model.summary()

# compile, fit
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer='adam',metrics=['mse','acc'])

model.fit([x_train,x_train], [y1_train,y2_train],epochs=10,batch_size=2,verbose=2)

# predict
loss = model.evaluate([x_train, x_train],[y1_train,y2_train])
print(f'loss : {loss}')

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred, x2_pred])
print(f'y_pred : {y_pred}')