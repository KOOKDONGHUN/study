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
def build_model(n_hidden=1,n_neurons=30,lr=3e-3,input_shape=[8], activation='relu'):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons,activation=activation))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(loss='mse', optimizer=optimizer)

    return model

# fit
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
keras_reg.fit(x_train,y_train, epochs=100, validation_data=(x_valid,y_valid))

# evaluate, predict
mse_test = keras_reg.score(x_test,y_test)

x_new = x_test[:3]
y_pred = keras_reg.predict(x_new)