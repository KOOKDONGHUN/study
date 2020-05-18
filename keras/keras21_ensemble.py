#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).transpose()
y = np.array([range(711,811),range(711,811),range(100)]).transpose()

x2 = np.array([range(101,201),range(411,511),range(100,200)]).transpose()
y2 = np.array([range(501,601),range(711,811),range(100)]).transpose()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split( 
    x,y,random_state = 66, shuffle=True,
    train_size=0.95)

x2_train,x2_test,y2_train,y2_test = train_test_split( 
    x2,y2,random_state = 66, shuffle=True,
    train_size=0.95)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#함수형 모델1
input1 = Input(shape=(3,)) #행 무시, 열 우선
dense1 = Dense(5,activation='relu',name='m1_d1')(input1)
dense1 = Dense(100,activation='relu',name='m1_d2')(dense1)
dense1 = Dense(50,activation='relu',name='m1_d3')(dense1)
dense1 = Dense(41,activation='relu',name='m1_d4')(dense1)
# output1 = Dense(3)(dense1)
# model = Model(inputs=input1, outputs=output1)

#함수형 모델2
input2 = Input(shape=(3,)) #행 무시, 열 우선
dense2 = Dense(5,activation='relu',name='m2_d1')(input2)
dense2 = Dense(150,activation='relu',name='m2_d2')(dense2)
dense2 = Dense(100,activation='relu',name='m2_d3')(dense2)
dense2 = Dense(50,activation='relu',name='m2_d4')(dense2)
# output2 = Dense(3)(dense1)

from keras.layers.merge import concatenate # concatenate 사전적 의미 : 사슬같이 있다 ( 단순병합 가장 기본적인 앙상블방법 )
merge1 = concatenate([dense1,dense2])
midel1 = Dense(92,name='e1_d1')(merge1)
midel1 = Dense(84,name='e1_d2')(midel1)
midel1 = Dense(55,name='e1_d3')(midel1)

#---------------output모델 구성 ----------------
output1 = Dense(21,name='m1_e1_d1')(midel1)
output1_2 = Dense(30,name='m1_e1_d2')(output1)
output1_3 = Dense(3,name='m1_e1_outpput')(output1_2)

output2 = Dense(30,name='m2_e1_d1')(midel1)
output2_2 = Dense(30,name='m2_e1_d2')(output2)
output2_3 = Dense(3,name='m2_e1_outpput')(output2_2)

model = Model(inputs=[input1,input2],
              outputs=[output1_3,output2_3])

model.summary(line_length=None,
              positions=None,
              print_fn=None)


#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit([x_train,x2_train],
          [y_train,y2_train],
          epochs=100,
          batch_size=2,
          validation_split=0.3,
          verbose=3) 


#4. 평가, 예측
li = []

li.append(model.evaluate([x_test,x2_test],
                          [y_test,y2_test],
                          batch_size=2))

print(li)
print("앙상블의 loss : ",li[0][0])
print("model1 loss : ",li[0][1])
print("model1 mse : ",li[0][2])
print("model2 loss : ",li[0][3])
print("model2 mse : ",li[0][4])


y_predict = model.predict([x_test,x2_test])
print(y_test,"\n\n",y2_test)
print("\n",y_predict[0][:],"\n\n",y_predict[1][:])

'''
#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test,y_predict))

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2 : ",r2_y_predict)

"""

 # Question

 # Note

 # homework

 """
 '''