# activation 넣어서 107번 파일에 추가하고 튜닝하기
# 100번을 카피해서 lr을 넣고 튠하시오.

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, LSTM,Dropout, Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam,RMSprop,SGD,Adadelta,Nadam
import tensorflow as tf
from keras.activations import selu,elu
# from tensorflow.nn import leaky_relu

leaky_relu = tf.nn.leaky_relu()

#1 데이터
print(np.linspace(0.1,0.5,5))
(x_train, y_train), (x_test,y_test) = mnist.load_data()


# x_train = x_train.reshape(x_train.shape[0],28,28,1).astype("float")/255.
# x_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float")/255.
x_train = x_train.reshape(x_train.shape[0],-1).astype("float")/255.
x_test = x_test.reshape(x_test.shape[0],-1).astype("float")/255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape,x_test.shape)

print(y_train.shape)
print(y_train)

# def build_model(drop=0.5,opt=None,lr=None,act=None):
def build_model(drop,opt,lr,act):
    opt = opt(lr=lr)
    inputs = Input(shape=(x_train.shape[1],), name='input')
    x = Dense(20,name='hidden1', activation=act)(inputs)
    x = Dropout(drop)(x)

    x = Dense(256, activation=act,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=act,name='hidden3')(x)
    x = Dropout(drop)(x)
    output1 = Dense(10,name='output',activation='softmax')(x)
    model = Model(inputs=inputs,outputs=output1)
    model.compile(optimizer=opt,metrics=['acc'],loss="categorical_crossentropy")


    return model

def create_hyper():
    batches = [64,128]# ,30,40,50]
    optimizer = [Adam,RMSprop]#,SGD,Adadelta,Nadam]
    droptout = list(np.linspace(0.1,0.5,5))
    lr = list(np.linspace(0.001,0.1,10))
    act = [leaky_relu, selu, elu]
    epoch = [10,15]
    splt = [0.1,0.2]

    return{"batch_size" : batches,'opt':optimizer ,"drop": droptout,'lr' : lr, 'act' : act, 'epochs' : epoch, 'validation_split' : splt}

model = KerasClassifier(build_fn=build_model, verbose = 1)
hyperparameters = create_hyper()

search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters,n_iter=1,cv=None,n_jobs=1)
search.fit(x_train,y_train)
# print(search.estimator.fit(x_test,y_test))
pred = search.predict(x_test)
print(pred)
print(y_train)

print("shape",y_test.shape)
print("shape",pred.shape)

try :
    pred = np.argmax(pred)
    score = accuracy_score(y_test,pred)
    print(score)
except :
    pass

# score = search.score(x_test)
print(search.best_params_)
