from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 탠서플로의 원핫 인코딩
# aaa = tf.one_hot(y, ???)
seed = 77
tf.set_random_seed(seed)

x_data, y_data = load_iris(return_X_y=True)

x_data = np.array(x_data,dtype=np.float32)

sess = tf.Session()
y_data = tf.one_hot(np.array(y_data,dtype=np.float32),depth=3,on_value=1,off_value=0).eval(session=sess)
sess.close()

x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=seed)

print(x_data.shape)
print(y_data.shape)

x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3

x = tf.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.placeholder(tf.float32, shape=[None, y_col_num])

w = tf.Variable(tf.random_normal([x_col_num, y_col_num]), name = 'weight')
b = tf.Variable(tf.random_normal([1, y_col_num]), name = 'bias') # y_col_num

h = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss) # 어떻게 쓰는지 어떻게 계산하는지 지금은 일단 쓰고 시간이 많을때 꼭!!! 공부하라 경사하강법

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        _, _, cost_val = sess.run([h, opt, loss], feed_dict={ x: x_data, y: y_data})

        if i % 200 == 0 :
            print( i , cost_val)
    
    pred = sess.run(h, feed_dict={x:x_test}) # keras model.predict(x_test_data)
    pred = sess.run(tf.argmax(pred, 1)) # tf.argmax(a, 1) 안에 값들중에 가장 큰 값의 인덱스를 표시하라
    # pred = pred.reshape(-1,1)
    print(pred)

    y_test = sess.run(tf.argmax(y_test,1))
    print(y_test)

    acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(pred, y_test),tf.float32))
    acc = sess.run(acc)
    print(acc)