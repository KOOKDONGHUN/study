'''p388, p389'''

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

# model
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs) # 표준매개변수 처리?? (예를 들면, name) ??
        self.h1 = keras.layers.Dense(units,activation=activation)
        self.h2 = keras.layers.Dense(units,activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1) # 보조 출력인가?

    def call(self,inputs):
        input_A, input_B = inputs
        h1 = self.h1(input_B)
        h2 = self.h2(h1)

        concat = keras.layers.concatenate([input_A,input_B])

        main_output = self.main_output(concat)
        aux_output = self.aux_output(h2)

        return main_output, aux_output

model = WideAndDeepModel() # 함수형과 비슷하지만 Input클래스의 객체를 만들필요 없다 // 대신 call() 메서드의 input 매개변수를 사용한다 


# compile, fit
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print(f'{epoch}\tval/train : {logs["val_loss"]/logs["loss"]}\n')

model.compile(loss=['mse','mse'], loss_weights=[0.9, 0.1], optimizer='adam')
hist = model.fit([x_train[:, :5],x_train[:, 2:]],[y_train,y_train], epochs=20,
                  validation_data=([x_valid[:, :5],x_valid[:, 2:]],[y_valid,y_valid]),
                  callbacks=[PrintValTrainRatioCallback()])

mse_test = model.evaluate([x_test[:, :5],x_test[:, 2:]], [y_test, y_test])
print(mse_test)

# predict
x_new_A, x_new_B = x_test[:, :5],x_test[:, 2:]
y_pred = model.predict([x_new_A, x_new_B]) # 이렇게 데이터 입력할때 (), [] error나는데 책에서는 그냥 () 쓴다 뭐지
print(y_pred)