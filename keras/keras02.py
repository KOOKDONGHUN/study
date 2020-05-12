import numpy as np
# 데이터 생성 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5,input_dim = 1,activation='relu'))
model.add(Dense(3))
model.add(Dense(2))
"""model.add(Dense(720))
model.add(Dense(100))
model.add(Dense(350))
model.add(Dense(10))"""
model.add(Dense(1,activation='relu'))

model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #        ※바이어스와 바이어스관계는 입력을 주거나 받지 않는다. 
=================================================================
                                                                                    인풋   바이어스

dense_1 (Dense)              (None, 5)                 10         --> 인풋 레이어에서 첫번째 히든레이어 | 총 인풋(입력 1개 + 바이어스 1개) * 출력 5개  = 10개
                                                                      노드n 노드n 노드n 노드n 노드n    바이어스 
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18         --> 첫번째 히든레이어에서 두번째 히든 레이어 | 총 인풋(입력5개 + 바이어스 1개) * 출력3개 = 18개
                                                                            노드n 노드n 노드n    바이어스
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8          --> 두번째 히든레이어에서 세번째 히든 레이어 | 총 인풋(입력3개 + 바이어스 1개) * 출력2개 = 8개
                                                                                노드n 노드n 바이어스
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3          --> 세번째 히든레이어에서 아웃풋 레이어 | 총 인풋(입력2개 + 바이어스 1개) * 출력1개 = 3개

                                                                                    아웃풋
=================================================================
Total params: 39
Trainable params: 39
Non-trainable params: 0
_________________________________________________________________
"""

model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=140, batch_size=3, validation_data=(x_train, y_train))
los, acc = model.evaluate(x_test, y_test, batch_size =3)

print("loss : " ,los )
print("acc : " ,acc )

output = (model.predict(x_test))
print(output)