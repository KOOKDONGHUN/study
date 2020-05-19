#1. 데이터
import numpy as np
x = np.array([range(1,101),range(301,401)]).transpose()

y = np.array([range(711,811),range(611,711)]).transpose()
y2 = np.array([range(101,201),range(411,511)]).transpose()

################## ####
#여기서부터 수정하시오 #
################## ####

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test,y2_train,y2_test = train_test_split( 
    # x,y,y2, shuffle=False,
    x,y,y2,random_state = 66, shuffle=True,
    train_size=0.95)

# 테스트셋이 잘 나누어 졌는지 확인 하기 위한 프린트문 
# print("\nx_train\n",x_train)
# print("\nx_test\n",x_test)
# print("\ny_train\n",y_train)
# print("\ny_test\n",y_test)
# print("\ny2_train\n",y2_train)
# print("\ny2_test\n",y2_test)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# 함수형 모델1
input1 = Input(shape=(2,)) #행 무시, 열 우선
dense1 = Dense(5,activation='relu',name='m1_d1')(input1)
dense1 = Dense(300,activation='relu',name='m1_d2')(dense1)
dense1 = Dense(300,activation='relu',name='m1_d3')(dense1)
dense1 = Dense(300,activation='relu',name='m1_d4')(dense1)

# from keras.layers.merge import concatenate # concatenate 사전적 의미 : 사슬같이 있다 ( 단순병합 가장 기본적인 앙상블방법 )
# merge1 = concatenate([dense1,dense1]) # 될까? 이게 가능하면 다르긴하지 가능하긴한데 2개의 모델로 인식은 안되는듯?
# merge1 = concatenate([dense1]) # 이것도 될까? -> 에러남 안됨 concatenatte에는 2개이상의 모델이 들어가야함

#근데 이렇게 하면 결국 그냥 순차 모델에 그냥 아웃풋만 2개 한거자나... 뭐가달라? -> 안쓰면 되는구나..
# midel1 = Dense(70,name='mid_d5')(merge1)
# midel1 = Dense(100,name='mid_d5')(dense1)
# midel1 = Dense(100,name='mid_d6')(midel1)
# midel1 = Dense(100,name='mid_d7')(midel1)

#---------------output모델 구성 ----------------#
# output1 = Dense(60,name='out_d8')(midel1)
output1 = Dense(60,name='out_d8')(dense1)
output1 = Dense(60,name='out_d9')(output1)
output1 = Dense(2,name='out_outpput1')(output1)

# output2 = Dense(60,name='out2_d8')(midel1)
output2 = Dense(60,name='out2_d8')(dense1)
output2 = Dense(60,name='out2_d9')(output2)
output2 = Dense(2,name='out2_outpput2')(output2)

# 함수형 모델의 선언
model = Model(inputs=input1,
              outputs=[output1,output2])

model.summary()

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 

from keras.callbacks import EarlyStopping
# 이미 일찍 멈추기 시작했다는것은 안좋아지기 시작했다는 의미 이므로 시점만 찾고 수작없으로 epochs을 조정하는게 더 좋다. -Y.YS-
els = EarlyStopping(monitor='loss', patience=3, mode='auto') # patience=10 -> loss의 값에서 변경을 자동으로 감지하고 10번정도 후에 멈춘다.

model.fit(x_train,
          [y_train,y2_train],
          epochs=100,
          batch_size=4,
          callbacks=[els],
          validation_split=0.3,
          verbose=2)


#4. 평가, 예측
li = model.evaluate(x_test,
                    [y_test,y2_test],
                    batch_size=4)

y1_predict = model.predict(x_test) # 그냥 리스트로 받아짐 
# y1_predict = model.predict(x_test) # 그냥 리스트로 받아짐 
# print(type(y1_predict))
# y1_predict = model.predict(x_test) # 그냥 리스트로 받아짐 
# y1_predict = np.array(y1_predict) # 변환해도 안됨 어렵네 뭐가 문제일까... -> 되는건데 내가 함수명이랑 같은걸 변수명으로 사용해서 함수가 사라짐 매우 장애인같은 실수를 함 
# print(type(y1_predict))

print(y_test)
print(y1_predict[0])
print(y2_test)
print(y1_predict[1])

#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test,y1_predict[0])
print("RMSE : ", rmse)
rmse2 = RMSE(y2_test,y1_predict[1])
print("RMSE2 : ", rmse2)

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score

def r2_y_predict(y_test,y_predict):
    return r2_score(y_test,y_predict)

r2 = r2_y_predict(y_test,y1_predict[0])
print("r2 : ",r2)

r2_2 = r2_y_predict(y_test,y1_predict[1])
print("r2_2 : ",r2_2)


"""

 # Question

 # Note

 # homework

 """
