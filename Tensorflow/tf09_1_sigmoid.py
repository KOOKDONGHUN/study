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

# cost = tf.reduce_mean(tf.square(h - y))
cost = -tf.reduce_mean( y * tf.log(h) + (1-y) * tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = opt.minimize(cost)

predicted = tf.compat.v1.cast(h > 0.5, tf.float32) # argmax 같은거인듯? 탠서플로에서 시그모이드나 소프트 맥스일경우 이런식의 라인을 추가 해야하는듯 하다
acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(predicted, y),tf.float32)) # cast에 대해 알아보자

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x : x_data, y : y_data})

        if step % 200 == 0:
            print(step, cost_val)
    
    h, a, c = sess.run([h, predicted, acc], feed_dict={x : x_data, y : y_data}) ## 딕 넣는게 한번 런 하면 된거 아닌가?

    print(f'\nh : {h}\npred : {a}\nacc : {c}')

sess.close()

# 딥러닝에서 딥이 빠진 러닝으로 activation은 Linear
# 텐서플로를 이용한 단순 선형 회기 모델링 