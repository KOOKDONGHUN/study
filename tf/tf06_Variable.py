import tensorflow as tf

tf.compat.v1.set_random_seed(777)

x_train = tf.compat.v1.placeholder(tf.float32)
y_train = tf.compat.v1.placeholder(tf.float32)

w = tf.Variable(0.7,tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(w)
print(aaa)
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = w.eval()
sess.close()
print(bbb)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = w.eval(session=sess)
print(ccc)
sess.close()
