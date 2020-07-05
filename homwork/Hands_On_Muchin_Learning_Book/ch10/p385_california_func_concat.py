from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras

housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split(housing['data'], housing['target'])

x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)#.astype(float)
x_valid = scaler.transform(x_valid)#.astype(float)
x_test = scaler.transform(x_test)#.astype(float)

input_ = keras.layers.Input(shape=x_train.shape[1:])
h1 = keras.layers.Dense(30,activation='relu')(input_)
h2 = keras.layers.Dense(30, activation='relu')(h1)
concat = keras.layers.Concatenate()([input_,h2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_],outputs=[output])

# model.compile(loss='mean_squared_error',optimizer='SGD') // lr가 커서 폭발함
# nan나오는 이유 // 완벽한데이터 한정 그라디언트 폭발이라고함 // 만약 데이터 자체가 불안정하다면 nan이 나오는 결과를 초래할 수 있다.
model.compile(loss='mean_squared_error',optimizer='adam')
hist = model.fit(x_train,y_train, epochs=20, validation_data=(x_valid,y_valid))

mse_test = model.evaluate(x_test, y_test)
print(mse_test)

x_new = x_test[:3]
y_pred = model.predict(x_new)
print(y_pred)