import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential, Model
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, Dropout, Flatten,Input
from keras.layers import MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor # 항상 분류와 회기가 존재한다 ㅎ
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

# 1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train,y_test = train_test_split(x,y,random_state=43,shuffle=True,test_size=0.2)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 2. model 모델 자체를 진짜 함수로 만든다?

def build_model(drop=0.1, optimizer='adam'):
    input1 = Input(shape=(4,),name='input1')
    x = Dense(512, activation='relu', name='hidden1')(input1)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    output = Dense(3,activation='softmax',name='output')(x)
    model = Model(inputs=input1,outputs=output)
    model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batchs = [16,32,64]
    optimizer = ['rmsprop','adam', 'adadelta']
    dropout = [0.5,0.6,0.7] # 리스트 형식만 가능 넘파이 안됨
    
    # return {'model__batch_size' : batchs,'model__optimizer' : optimizer, 'model__drop': dropout}
    return {'kerasclassifier__batch_size' : batchs,'kerasclassifier__optimizer' : optimizer, 'kerasclassifier__drop': dropout}

# 케라스를 그냥 쓰면 안된다? sklearn형식으로 wrap한다 gridsearch, randomsearch를 사용하기 위해서 

model = KerasClassifier(build_fn=build_model) # 사이킷런에서 쓸수 있도록 wrapping했다 

hyperparameters = create_hyperparameters()

# pipe = Pipeline([('scaler', MinMaxScaler()),('model',model)])
pipe = make_pipeline(MinMaxScaler(),KerasClassifier(build_fn=build_model))

# print(f"pipe.get_params():{pipe.get_params()}")

model = RandomizedSearchCV(pipe, hyperparameters, cv=2)  # -> 첫번째 인자에 모델이 들어가고 파라미터, kfold

try :
    model.fit(x_train,y_train)
except:
    print("Error")

print(model.best_estimator_)
print(model.best_params_)
print(model.score(x_test,y_test))