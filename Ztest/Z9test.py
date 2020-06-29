# 1.데이터 생성 
import numpy as np
from sklearn.model_selection import KFold

# y = wx + b 라는 함수에 들어갈 x 값과 그에 맞는 y값
# w는 가중치(기울기), b는 baias(y절편)
# np.array를 쓰면 x에 있는 리스트가 행렬의 형태로 변환되고 x안에 있는 값들을 행렬연산이 가능하도록 해준다.

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

skf = KFold(n_splits=4, shuffle=True)

for train, validation in skf.split(x,y):
    print(train, validation)
    print()
    print(x[train])
    print(x[validation])
    # print()
    print(y[train])
    print(y[validation])


''' 
# 2.모델구성
from keras.models import Sequential # 층의 구성을 시작하는 인풋(인풋 레이어)에서 아웃풋(아웃풋 레이어)으로 바로갈수 없는건 아니지만 그렇다면 딥러닝이 아님 
                                    # 떄문에 중간(deep 또는 히든레이어)을 거쳐 간다는 의미 계층구조
from keras.layers import Dense # 중간(딥)을 Dense라는 모듈(함수)를 사용하여 각 층을 설계 하겠다.

model = Sequential() # Sequential()이라는 클래스를 model이라는 이름으로 사용하겠다 (객체생성)
                     # model이라는 빵틀에 Sequential()을 넣어서 빵을 만들겠다.
model.add(Dense(1,input_dim = 1,activation='relu'))


# 3. 훈련
from keras.callbacks import EarlyStopping
els = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss = 'mean_squared_error', optimizer='adam',metrics=['accuracy']) #내가 설계한 모델이 머신(컴퓨터 예를 들면 사람)이 알아 먹을수 있도록 해주는 작업 
# model.compile(loss = 'mean_squared_error', optimizer='adam')

model.fit(x,y,epochs=500,batch_size=1,callbacks=[els]) # 모델에 학습을 시키는 과정


# 4. 평가 예측
# los = model.evaluate(x, y, batch_size =1)
los, acc = model.evaluate(x, y, batch_size =1) # 머신이 내가 설계한 모델로 학습했고 얼마나 학습이 
                                               # 잘 됐는가 평가 해서 나오는 값을 los, acc (변수)에 저장
                                               # 평가했을때 2가지가 나오는 것은 metrics=['accuracy']이 문장 때문이다. 기본적으로 loss는 나옴 

print("loss : " ,los )
print("acc : " ,acc ) '''