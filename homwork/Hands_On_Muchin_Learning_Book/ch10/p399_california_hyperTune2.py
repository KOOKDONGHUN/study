from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
import numpy as np

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
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mse', optimizer=optimizer)

    return model

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden' : [0, 1, 2, 3],
    'n_neurons' : np.arange(1, 100),
    'lr' : reciprocal(3e-4,3e-2) # 역수?
}

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
rnd__search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=None)

# fit
rnd__search_cv.fit(x_train,y_train, epochs=2, validation_data=(x_valid,y_valid))

# evaluate, predict
mse_test = rnd__search_cv.best_score_
print(mse_test)

best_param = rnd__search_cv.best_params_
print(best_param)

best_param = rnd__search_cv.best_estimator_
print(best_param)

x_new = x_test[:3]
y_pred = rnd__search_cv.predict(x_new)
print(y_pred)

''' mse_test = rnd__search_cv.best_score_(x_test.astype(float),y_test.astype(float))
mse_test = rnd__search_cv.best_score_(x_test,y_test) # TypeError: 'numpy.float64' object is not callable '''