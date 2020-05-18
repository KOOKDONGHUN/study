#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411)]).transpose()
x2 = np.array([range(101,201),range(411,511)]).transpose()

y = np.array([range(711,811),range(711,811)]).transpose()
y2 = np.array([range(501,601),range(711,811)]).transpose()
y3 = np.array([range(411,511),range(611,711)]).transpose()

################## ####
#여기서부터 수정하시오 #
################## ####

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    # x,y, shuffle=False,
    train_size=0.95)

x2_train,x2_test,y2_train,y2_test = train_test_split( 
    x2,y2,random_state = 66, shuffle=True,
    # x2,y2, shuffle=False,
    train_size=0.95)

y3_train,y3_test = train_test_split(
    y3,random_state = 66, shuffle=True,
    # y3, shuffle=False,
    train_size=0.95)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#함수형 모델1
input1 = Input(shape=(2,)) #행 무시, 열 우선
dense1 = Dense(5,activation='relu',name='m1_d1')(input1)
dense1 = Dense(150,activation='relu',name='m1_d2')(dense1)
dense1 = Dense(50,activation='relu',name='m1_d3')(dense1)
dense1 = Dense(41,activation='relu',name='m1_d4')(dense1)
# output1 = Dense(3)(dense1)
# model = Model(inputs=input1, outputs=output1)

#함수형 모델2
input2 = Input(shape=(2,)) #행 무시, 열 우선
dense2 = Dense(5,activation='relu',name='m2_d1')(input2)
dense2 = Dense(150,activation='relu',name='m2_d2')(dense2)
dense2 = Dense(100,activation='relu',name='m2_d3')(dense2)
dense2 = Dense(50,activation='relu',name='m2_d4')(dense2)
# output2 = Dense(3)(dense1)

from keras.layers.merge import concatenate # concatenate 사전적 의미 : 사슬같이 있다 ( 단순병합 가장 기본적인 앙상블방법 )
merge1 = concatenate([dense1,dense2])
midel1 = Dense(110,name='e1_d1')(merge1)
midel1 = Dense(100,name='e1_d2')(midel1)
midel1 = Dense(110,name='e1_d3')(midel1)

#---------------output모델 구성 ----------------#
output1 = Dense(21,name='m1_e1_d1')(midel1)
output1_2 = Dense(30,name='m1_e1_d2')(output1)
# output1_3 = Dense(3,name='m1_e1_outpput')(output1_2)...여기를 고치는걸 못했음 ㅅ..
output1_3 = Dense(2,name='m1_e1_outpput')(output1_2)

output2 = Dense(30,name='m2_e1_d1')(midel1)
output2_2 = Dense(30,name='m2_e1_d2')(output2)
# output2_3 = Dense(3,name='m2_e1_outpput')(output2_2) ...여기를 고치는걸 못했음 ㅅ..
output2_3 = Dense(2,name='m2_e1_outpput')(output2_2)

output3 = Dense(30,name='m_e1_d1')(midel1)
output3_2 = Dense(30,name='m_e1_d2')(output3)
# output3_3 = Dense(3,name='m_e1_outpput')(output3_2)...여기를 고치는걸 못했음 ㅅ..
output3_3 = Dense(2,name='m_e1_outpput')(output3_2)

# 함수형 모델의 선언
model = Model(inputs=[input1,input2],
              outputs=[output1_3,output2_3,output3_3])

# model.summary()


#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([x_train,x2_train],
          [y_train,y2_train,y3_train],
          epochs=150,
          batch_size=3,
          validation_split=0.3,
          verbose=2)


#4. 평가, 예측
li = model.evaluate([x_test,x2_test],
                    [y_test,y2_test,y3_test],
                    batch_size=2)

print(li)
print("앙상블의 loss : ",li[0])
print("model1 loss : ",li[1])
print("model2 loss : ",li[2])
print("model1 mse : ",li[3])
print("model2 mse : ",li[4])

# m1과 m2의 예측값 
y1_predict, y2_predict, y3_predict = model.predict([x_test,x2_test])

print(y_test,"\n\n",y2_test)
print("\n",y1_predict,"\n\n",y2_predict)

#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# m1,m2,e1  모델1, 모델2, 앙상블모델1의 RMSE
m1_RMSE = RMSE(y_test,y1_predict)
m2_RMSE = RMSE(y2_test,y2_predict)
m_RMSE = RMSE(y3_test,y3_predict) #추가 
e1_RMSE = (m1_RMSE + m2_RMSE + m_RMSE)/3

# print("m1_RMSE : ", m1_RMSE)
# print("m2_RMSE : ", m2_RMSE)
# print("m_RMSE : ", m_RMSE) #추가 
print("e1_RMSE : ", e1_RMSE)


#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score

def r2_y_predict(y_test,y_predict):
    return r2_score(y_test,y_predict)

# m1,m2,e1  모델1, 모델2, 앙상블모델1의 r2
m1_r2 = r2_y_predict(y_test,y1_predict)
m2_r2 = r2_y_predict(y2_test,y2_predict)
m_r2 = r2_y_predict(y3_test,y3_predict)  #추가 
e1_r2 = (m1_r2 + m2_r2 + m_r2)/3

# print("m1_r2 : ",m1_r2)
# print("m2_r2 : ",m2_r2)
# print("m_r2 : ",m_r2)      #추가 
print("e1_r2 : ",e1_r2)


"""

 # Question

 # Note

    단순모델일때 순차 모델
    여러가지 모델을 합치고 싶을때 함수 모델

    100,6은 데이터 셋을 합쳐야한다 
    but 합치지 않고 해도 상관없음 각자의 편의대로 하면됨 

 # homework

 """
