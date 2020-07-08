import tensorflow as tf

tf.set_random_seed(777)


x_train = tf.placeholder(dtype=tf.float32)
y_train = tf.placeholder(dtype=tf.float32)

x = [1,2,3]
y = [3,5,7]

# feed_dict = {x_train:x, y_train:y}


W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))

hypothesis = x_train * W + b       

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) 

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    sess.run(hypothesis, feed_dict={x_train: [1, 2, 3]})
    sess.run(cost, feed_dict={y_train : [3, 5, 7] })


    for step in range(2001):    
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  

        if step % 20 == 0:  
            print(step, cost_val, W_val, b_val)