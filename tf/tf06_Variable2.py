#hyporthesis

import tensorflow as tf

tf.compat.v1.set_random_seed(777)

x = [1., 2., 3.]
w = tf.Variable([0.3],tf.float32, name='weight')
b = tf.Variable([1.],tf.float32, name='bias')

hypothesis = w*x + b

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# aaa = sess.run(w)
# aaa = sess.run(b)
aaa = sess.run(hypothesis)
print(aaa)
sess.close()

# sess = tf.compat.v1.InteractiveSession()
# sess.run(tf.compat.v1.global_variables_initializer())
# bbb = w.eval()
# sess.close()
# print(bbb)

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# ccc = w.eval(session=sess)
# print(ccc)
# sess.close()
