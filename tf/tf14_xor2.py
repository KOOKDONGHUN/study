import tensorflow as tf
import numpy as np
seed = 777
tf.set_random_seed(seed)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# x, y, w, b, h, cost, train
# sigmoid, predict, acc

x_col_num = x_data.shape[1]
y_col_num = y_data.shape[1]

x = tf.placeholder(tf.float32, shape=[None,x_col_num])
y = tf.placeholder(tf.float32, shape=[None,y_col_num])

w1 = tf.Variable(tf.random_normal([x_col_num, 16]), name = 'weight1') # 다음 레이어에 100개를 전달? 노드의 갯수와 동일하다고 봐도 무방?
b1 = tf.Variable(tf.random_normal([16]), name = 'bias1') # 자연스럽게 100을따라감 ?? 왜?
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([16, 8]), name='weight2')
b2 = tf.Variable(tf.random_normal([8]),name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([8, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]),name = 'bias3')
h = tf.sigmoid(tf.matmul(layer2, w3) + b3)

# cost = tf.reduce_mean(tf.square(h - y))
cost = -tf.reduce_mean( y * tf.log(h) + (1-y) * tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=1)
train = opt.minimize(cost)

predicted = tf.compat.v1.cast(h > 0.5, tf.float32) 
acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(predicted, y),tf.float32)) 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x : x_data, y : y_data})

        if step % 200 == 0:
            print(step, cost_val)
    
    h, a, c = sess.run([h, predicted, acc], feed_dict={x : x_data, y : y_data}) 

    print(f'\nh : {h}\npred : {a}\nacc : {c}')

sess.close()
