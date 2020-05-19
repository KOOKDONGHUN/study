#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411),range(411,511)]).transpose()
x2 = np.array([range(711,811),range(711,811),range(511,611)]).transpose()

y = np.array([range(101,201),range(411,511),range(100)]).transpose()

# 데이터가 적으니까 데이터 나누기 한번에 안되는데? 
# x = np.array([range(1,11,range(31,41),range(41,51)]).transpose()
# x2 = np.array([range(71,81),range(71,81),range(51,61)]).transpose()
# y = np.array([range(11,21),range(41,51),range(10)]).transpose()

################## ####
#여기서부터 수정하시오 #
################## ####

from sklearn.model_selection import train_test_split


# x_train,x_test,y_train,y_test = train_test_split( 
#     x,y,random_state = 66, shuffle=True,
#     # x,y, shuffle=False,
#     train_size=0.95)
# x2_train,x2_test = train_test_split( 
#     x2,random_state = 66, shuffle=True,
#     # x,y, shuffle=False,
#     train_size=0.95)

x_train,x_test,x2_train,x2_test,y_train,y_test = train_test_split( 
    x,x2,y,random_state = 66, shuffle=True,
    # x,x2,y, shuffle=False,
    train_size=0.9)

# 테스트셋이 잘 나누어 졌는지 확인 하기 위한 프린트문 
# print("\nx_train\n",x_train)
# print("\nx_test\n",x_test)
# print("\nx2_train\n",x2_train)
# print("\nx2_test\n",x2_test)
# print("\ny_train\n",y_train)
# print("\ny_test\n",y_test)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# 함수형 모델1
input1 = Input(shape=(3,)) #행 무시, 열 우선
dense1 = Dense(5,activation='relu',name='m1_d1')(input1)
dense1 = Dense(100,activation='relu',name='m1_d2')(dense1)
dense1 = Dense(30,activation='relu',name='m1_d3')(dense1)
dense1 = Dense(31,activation='relu',name='m1_d4')(dense1)
# output1 = Dense(3)(dense1)
# model = Model(inputs=input1, outputs=output1)

# 함수형 모델2
input2 = Input(shape=(3,)) #행 무시, 열 우선
dense2 = Dense(5,activation='relu',name='m2_d1')(input2)
dense2 = Dense(100,activation='relu',name='m2_d2')(dense2)
dense2 = Dense(40,activation='relu',name='m2_d3')(dense2)
dense2 = Dense(41,activation='relu',name='m2_d4')(dense2)
# output2 = Dense(3)(dense1)

# 모델1 + 모델2
from keras.layers.merge import concatenate # concatenate 사전적 의미 : 사슬같이 있다 ( 단순병합 가장 기본적인 앙상블방법 )
merge1 = concatenate([dense1,dense2])
midel1 = Dense(70,name='e1_d5')(merge1)
midel1 = Dense(70,name='e1_d6')(midel1)
midel1 = Dense(70,name='e1_d7')(midel1)

#---------------output모델 구성 ----------------#
output1 = Dense(60,name='e1_d8')(midel1)
output1_2 = Dense(60,name='e1_d9')(output1)
output1_3 = Dense(3,name='e1_outpput')(output1_2)

# 함수형 모델의 선언
model = Model(inputs=[input1,input2],
              outputs=output1_3)

# model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([x_train,x2_train],
          y_train,
          epochs=80,
          batch_size=3,
          validation_split=0.3,
          callbacks=[els],
          verbose=2)


#4. 평가, 예측
li = model.evaluate([x_test,x2_test],
                    y_test,
                    batch_size=3)

y1_predict = model.predict([x_test,x2_test])

print(y_test)
print(y1_predict)


#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

RMSE = RMSE(y_test,y1_predict)
print("RMSE : ", RMSE)

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score

def r2_y_predict(y_test,y_predict):
    return r2_score(y_test,y_predict)

r2 = r2_y_predict(y_test,y1_predict)
print("r2 : ",r2)


"""

 # Question

 # Note
    
    p.156 까지 내용 복습 하기 

    단순모델일때 순차 모델
    여러가지 모델을 합치고 싶을때 함수 모델

    100,3의 비슷한 데이터 2개를 순차 모델로 할때 합쳐서 하면됨
    but 합치지 않고 해도(함수모델) 상관없음 각자의 편의대로 하면됨 

 # homework

 """
