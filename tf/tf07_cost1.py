import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [3.,5.,7.]

w = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_hist = []
cost_hist = []

with tf.compat.v1.Session() as sess:
    # sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(-30,50):
        curr_w = i * 0.1 # 그림의 간격
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_hist.append(curr_w)
        cost_hist.append(curr_cost)

plt.plot(w_hist,cost_hist)
plt.show()