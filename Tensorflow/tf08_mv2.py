import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(77)

# 칼럼별로 가중치가 있어야한다
x_data = [[ 73., 53., 65.],
          [ 92., 98., 11 ],
          [89., 31., 33.],
          [99., 33., 100.],
          [17., 66., 79.]]

y_data = [[152.], 
          [185.], 
          [180.], 
          [205.], 
          [142.]]

x = tf.placeholder(tf.float32, shape=[None,3]) # shape를 해주는 이유가 딱히 있는건가
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight') # [3, 마음대로] 3인 이유는 x칼럼의 개수만큼 행렬연산이기 때문
b = tf.Variable(tf.random_normal([1]), name = 'bias') # 더하기 이기때문에 1

h = tf.matmul(x, w) + b # w*x + b // [5,3] * [3,1] = [5,1]  행렬연산 //
# h = x * w + b #  행렬연산이기 떄문에 안됨

cost = tf.reduce_mean(tf.square(h - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.000017)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x : x_data, y : y_data})

    if step % 20 == 0:
        print(step, 'cost : ', cost_val, '\n', h_val)

sess.close()