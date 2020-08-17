from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import warnings
 
warnings.filterwarnings("ignore")

# 탠서플로의 원핫 인코딩
# aaa = tf.one_hot(y, ???)
seed = 0
tf.compat.v1.set_random_seed(seed)

(x_data, y_data), (x_test, y_test) = cifar10.load_data()

print(x_data.shape)
print(y_data.shape)

sess = tf.compat.v1.Session()
y_data = tf.one_hot(y_data,depth=10,on_value=1,off_value=0).eval(session=sess)
y_test = tf.one_hot(y_test,depth=10,on_value=1,off_value=0).eval(session=sess)
sess.close()
y_data = y_data.reshape(-1,10)
y_test = y_test.reshape(-1,10)
print(y_data.shape)

x_data = x_data.astype('float32')/255.
print(x_data.shape)
x_test = x_test.astype('float32')/255.
print(x_test.shape)

x_col_num = x_data.shape[1] # 32
x_chanel = x_data.shape[3] # 3
y_col_num = y_data.shape[1] # 10

print(x_col_num)
print(y_col_num)
print(x_chanel)

lr =1e-2
epochs = 20
batch_size = 50
node_num = 2
total_batch = int(len(x_data) / batch_size)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_col_num])
x = tf.reshape(x , [-1, x_col_num, x_col_num, x_chanel])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])
drop_rate = tf.compat.v1.placeholder(tf.float32)

#### -- layers1
w1 = tf.get_variable('w1', shape=[3, 3, x_chanel, node_num**8])
l1 = tf.nn.conv2d(x, w1, strides=[1,2,2,1], padding='SAME')
print(f'l1 : {l1}')
l1 = tf.nn.selu(l1)
print(f'l1 : {l1}')
l1 = tf.nn.max_pool2d(l1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(f'l1 : {l1}')
print(f'l1.shape[3] : {l1.shape[3]}')
#### -- layers1

#### -- layers2
w2 = tf.get_variable('w2', shape=[3, 3, l1.shape[3], node_num**8]) # 3번째 shape 이전 노드의 개수
l2 = tf.nn.conv2d(l1, w2, strides=[1,2,2,1], padding='SAME')
l2 = tf.nn.selu(l2)
l2 = tf.nn.max_pool2d(l2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

l2 = tf.reshape(l2, [-1, l2.shape[1]*l2.shape[2]*l2.shape[3]]) # keras Flatten()
print(f'l2.shape[1] : {l2.shape[1]}')
#### -- layers2

## // 컨볼루션에는 bias연산을 알아서 해줌으로 따로 명시할 필요는 없다 ... ??
# is it True? -> 연산을 알아서 해준다기 보다는 합성곱 신경망 자체의 연산이 많은데
# 바이어스의 더하기 연산을 추가한다해서 큰 변화가 있을 수 있고 없을 수 있다.

#### -- layers3
w3 = tf.get_variable('w3', shape=[l2.shape[1], 10],
                      initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random.normal([10]))
h = tf.nn.softmax(tf.matmul(l2,w3)+b3)
#### -- layers3

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...
opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss) # 어떻게 쓰는지 어떻게 계산하는지 지금은 일단 쓰고 시간이 많을때 꼭!!! 공부하라 경사하강법

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        avg_cost = 0

        for i in range(total_batch):
            start_n = i*batch_size
            end_n = start_n + batch_size

            batch_xs, batch_ys = x_data[start_n:end_n,:], y_data[start_n: end_n, :]

            feed_d = {x : batch_xs, y : batch_ys, drop_rate : 0.5}
            c , _ = sess.run([loss, opt], feed_dict=feed_d)
            avg_cost += c / total_batch

        print(f'epoch : {step+1}\t cost : {avg_cost}')
    print('training finish !!')

    prediction = tf.equal(tf.arg_max(h, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(prediction, tf.float32)).eval(session=sess,feed_dict=feed_d)

    print(f'acc : {acc}')