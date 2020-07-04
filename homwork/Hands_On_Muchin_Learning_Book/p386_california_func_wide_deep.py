from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras

# data
housing = fetch_california_housing()

print(housing['data'].shape)
print(housing['target'].shape)

x_train_full, x_test, y_train_full, y_test = train_test_split(housing['data'], housing['target'])

x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# model // 함수형 모델에서 보조 출력을 쓰는이유? // 보조 출력 적용가능 최종 레이어를 거치지 않고 중간에 아웃풋을 내는 경우
input_A = keras.layers.Input(shape=[5],name="wide_input") # shape 입력 형식 [],() 상관없는듯

input_B = keras.layers.Input(shape=[6],name="deep_input")
h1 = keras.layers.Dense(30,activation='relu')(input_B)
h2 = keras.layers.Dense(30, activation='relu')(h1)

concat = keras.layers.Concatenate()([input_A,h2])

output = keras.layers.Dense(1,name="output")(concat)

model = keras.Model(inputs=[input_A,input_B],outputs=[output])

model.summary()

# compile, fit
model.compile(loss='mean_squared_error',optimizer='adam') # los_weight를 이용하여 보조출력의 경우 중요도를 낮출수 있다.
hist = model.fit([x_train[:, :5],x_train[:, 2:]],y_train, epochs=20, validation_data=([x_valid[:, :5],x_valid[:, 2:]],y_valid))

mse_test = model.evaluate([x_test[:, :5],x_test[:, 2:]], y_test)
print(mse_test)

# predict
x_new_A, x_new_B = x_test[:, :5],x_test[:, 2:]
y_pred = model.predict([x_new_A, x_new_B]) # 이렇게 데이터 입력할때 (), [] error나는데 책에서는 그냥 () 쓴다 뭐지
print(y_pred)