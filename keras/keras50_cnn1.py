from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

'''cnn은 입력이 4차원'''

model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(20,20,1))) # 1 또는 3 : 흑백 또는 컬러 채널
                 # 가로,세로,명암
''' filter, kernel_size=(2,2)'''
model.add(Conv2D(7,(3,3)))
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Conv2D(5,(2,2)))
# model.add(Conv2D(5,(2,2),strides=2))
# model.add(Conv2D(5,(2,2),strides=2,padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(5,(2,2),padding='same',strides=2))
model.add(Flatten())
model.add(Dense(1))

model.summary()

'''과제3개 메일로 보내기

컨볼루션신경망 파라미터 갯수구하기 과제 

(아웃풋 노드의 개수 * 필터(2*2) * 인풋 갯수) + 아웃풋 노드의 갯수
'''