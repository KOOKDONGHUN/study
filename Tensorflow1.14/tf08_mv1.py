import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(77)

# 칼럼별로 가중치가 있어야한다
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(h - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x1 : x1_data, x2 : x2_data, x3 : x3_data, y : y_data})

    if step % 20 == 0:
        print(step, 'cost : ', cost_val, '\n', h_val)

sess.close()