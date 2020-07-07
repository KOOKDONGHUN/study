import tensorflow as tf

node1 = tf.constant(3)
node2 = tf.constant(4)
node3 = tf.constant(5)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a : 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a : [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a : 3, b : 4.5}))