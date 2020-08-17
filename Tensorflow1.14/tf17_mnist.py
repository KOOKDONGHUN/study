from keras.datasets import mnist
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

(x_data, y_data), (x_test, y_test) = mnist.load_data()

print(x_data.shape)
print(y_data.shape)


sess = tf.compat.v1.Session()
y_data = tf.one_hot(y_data,depth=10,on_value=1,off_value=0).eval(session=sess)
y_test = tf.one_hot(y_test,depth=10,on_value=1,off_value=0).eval(session=sess)
sess.close()
print(y_data.shape)

x_data = x_data.reshape(-1,x_data.shape[1]*x_data.shape[2]).astype('float32')/255.
print(x_data.shape)
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype('float32')/255.
print(x_test.shape)

x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3

print(x_col_num)
print(y_col_num)

lr =1e-2
epochs = 20
batch_size = 50
total_batch = int(len(x_data) / batch_size)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])
drop_rate = tf.compat.v1.placeholder(tf.float32)

#### -- layers1
w1 = tf.get_variable('w1', shape=[x_col_num, 256],
                      initializer=tf.contrib.layers.xavier_initializer()) # variable보다 진화된것? 둘의 차이  커널에 대한 초기화?
print(w1) # <tf.Variable 'w1:0' shape=(784, 256) dtype=float32_ref> // shape 확인
# b1 = tf.get_variable('b1', )
b1 = tf.Variable(tf.random.normal([256]))
print(b1) # <tf.Variable 'Variable:0' shape=(256,) dtype=float32_ref> // shape 확인
l1 = tf.nn.selu(tf.matmul(x,w1)+b1)
print(l1) # Tensor("Selu:0", shape=(?, 256), dtype=float32) // shape 확인
l1 = tf.nn.dropout(l1,keep_prob=drop_rate)
print(l1) # Tensor("dropout/mul_1:0", shape=(?, 256), dtype=float32) // shape 확인
#### -- layers1

#### -- layers2
w2 = tf.get_variable('w2', shape=[256, 256],
                      initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([256]))
l2 = tf.nn.selu(tf.matmul(l1,w2)+b2)
l2 = tf.nn.dropout(l2,keep_prob=drop_rate)
#### -- layers2

#### -- layers3
w3 = tf.get_variable('w3', shape=[256, 512],
                      initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([512]))
l3 = tf.nn.selu(tf.matmul(l2,w3)+b3)
l3 = tf.nn.dropout(l3,keep_prob=drop_rate)
#### -- layers3

#### -- layers4
w4 = tf.get_variable('w4', shape=[512, 256],
                      initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([256]))
l4 = tf.nn.selu(tf.matmul(l3,w4)+b4)
l4 = tf.nn.dropout(l4,keep_prob=drop_rate)
#### -- layers4

#### -- layers5
w5 = tf.get_variable('w5', shape=[256, 10],
                      initializer=tf.contrib.layers.xavier_initializer()) 
b5 = tf.Variable(tf.random.normal([10]))
h = tf.nn.softmax(tf.matmul(l4,w5)+b5)
#### -- layers5

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