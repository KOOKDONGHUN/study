from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor # 항상 분류와 회기가 존재한다 ㅎ
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0,index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0,index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0,index_col=0)

print('train.shape : ',train.shape) # 10000,75 : x_train, x_test
print('test.shape : ',test.shape) # 10000,71 : x_predict
print('submission.shape : ',submission.shape) # 10000, 4 : y_predict

# print(train.isnull().sum()) 

# train = train.interpolate() # ? 뭔법? -> 보간법//선형보간 # 값들 전체의 선을 그리고 그에 상응하는 값을 알아서 넣어준다? ?? 하나의 선을 긋고 ? 그럼 완전 간단한 예측 모델을 만든거네
# print(train.isnull().sum()) 
# print(train.isnull().any()) 
# 회기 

# for i in train.columns:
#     # print(i)
#     print(len(train[train[i].isnull()]))

# 보간법에서 앞뒤가 nan이면 채워지는 값도 nan

# train = train.fillna(train.mean(),axis=0)
# train = train.fillna(method='bfill')
# test = test.fillna(method='bfill')
train = train.fillna(train.mean(),axis=0)
test = train.fillna(test.mean(),axis=0)
# print()

# for i in train.columns:
#     # print(i)
#     print(len(train[train[i].isnull()]))

train = train.values
test = test.values

plt.figure(figsize=(71,71))
sns.heatmap(train,linewidths=0.1,vmax=0.5,linecolor='white',annot=True)
plt.show()

''' 
print(type(train))

x_data = train[:, :-4]
y_data = train[:, -4:]

print(x_data.shape)
print(y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,random_state=3,shuffle=True,test_size=0.2)

# mm = MinMaxScaler()
# x_train = mm.fit_transform(x_train)
# x_test = mm.transform(x_test)
# test = mm.transform(test)

# 2. model
def build_model(drop=0.4, optimizer='adam'):
    input1 = Input(shape=(71,),name='input1')
    x = Dense(256, activation='relu', name='hidden1')(input1)
    x = Dropout(drop)(x)
    x = Dense(256, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, name='hidden3')(x)
    output = Dense(4,activation='relu',name='output')(x)
    model = Model(inputs=input1,outputs=output)
    model.compile(optimizer=optimizer,metrics=['mae'],loss='mse')

    return model

def create_hyperparameters():
    batchs = [32, 64, 128]
    optimizer = ['rmsprop','adam', 'adadelta']
    dropout = np.linspace(0.4, 0.8, 4)
    
    return {'batch_size' : batchs,'optimizer' : optimizer, 'drop': dropout}

model = KerasRegressor(build_fn=build_model,verbose=1) # 사이킷런에서 쓸수 있도록 wrapping했다 

param = create_hyperparameters()

search = RandomizedSearchCV(model,param,cv=4)
try :
    search.fit(x_train,y_train)
except:
    print("Error !!")

# print(search.best_estimator_)
print(search.best_params_)
# print(search.score(x_test,y_test))

# 3. compile, fit
# search.compile(optimizer='adam',loss = 'mse', metrics = ['mae'])

# search.fit(x_train,y_train,epochs=30,batch_size=3,callbacks=[],verbose=2,validation_split=0.1)

# loss = search.evaluate(x_test,y_test)

# print(loss)

y_pred = search.predict(test)

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id') '''