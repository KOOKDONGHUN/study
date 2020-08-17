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

w = tf.Variable(tf.zeros([x_col_num, y_col_num]), name = 'weight') 
b = tf.Variable(tf.zeros([y_col_num]), name = 'bias') 

h = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(h - y))
cost = -tf.reduce_mean( y * tf.log(h) + (1-y) * tf.log(1-h))

opt = tf.train.GradientDescentOptimizer(learning_rate=1)
train = opt.minimize(cost)

predicted = tf.compat.v1.cast(h > 0.5, tf.float32) # argmax 같은거인듯? 탠서플로에서 시그모이드나 소프트 맥스일경우 이런식의 라인을 추가 해야하는듯 하다
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
