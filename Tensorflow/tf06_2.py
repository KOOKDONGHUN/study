'''
tf06_01.py를 카피해서
lr을 수정해서 연습
epoch가 2000이하
'''

import tensorflow as tf

# tf.set_random_seed(777)
tf.compat.v1.set_random_seed(1)

# x_train = [1,2,3]
# # y_train = [1, 2, 3]
# y_train = [3,5,7]

x_train = tf.compat.v1.placeholder(tf.float32)
y_train = tf.compat.v1.placeholder(tf.float32)

# x_train = tf.placeholder(tf.float32)
# y_train = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 초기화? 뭐라는거야
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))
# print(sess.run(b))

hypothesis = w * x_train + b # 이게 모델이다...

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # 경사하강법 인데 최소의 cost를 찾겠다 
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0945).minimize(cost) # 경사하강법 인데 최소의 cost를 찾겠다 

# train = tf.train.MomentumOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess: # 전체가 범위안에 포함된다? 이해안됨 안에있는 세션이 다 실행한다 ??
    # sess.run(tf.global_variables_initializer()) # 변수들을 초기화 
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(1001):
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict={x_train: [1, 2, 3], y_train : [3, 5, 7] })

        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)

    # predict
    print(sess.run(hypothesis, feed_dict={x_train : [4] }))
    print(sess.run(hypothesis, feed_dict={x_train : [5,6,7] }))