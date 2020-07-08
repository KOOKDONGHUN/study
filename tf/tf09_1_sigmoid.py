import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]])

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1],]

print(x_data.shape)

col_num = x_data.shape[1]
print(col_num)

x = tf.placeholder(tf.float32, shape=[None,col_num]) # shape를 해주는 이유가 딱히 있는건가
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([col_num, 1]), name = 'weight') # [3, 마음대로] 3인 이유는 x칼럼의 개수만큼 행렬연산이기 때문
b = tf.Variable(tf.random_normal([1]), name = 'bias') # 더하기 이기때문에 1

h = tf.sigmoid(tf.matmul(x, w) + b)

cost = tf.reduce_mean(tf.square(h - y))

cost = -tf.reduce_mean( y * tf.log(h) + (1-y) * tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.000038)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x : x_data, y : y_data})

    if step % 20 == 0:
        print(step, h_val, '\ncost : ', cost_val)

sess.close()

# 딥러닝에서 딥이 빠진 러닝으로 activation은 Linear
# 텐서플로를 이용한 단순 선형 회기 모델링 