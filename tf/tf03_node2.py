import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)

# node3 = tf.add_n(node1, node2, node3)
node4 = tf.add_n([node1, node2, node3])
print(f'node1 : {node1}\t node2 : {node2}')
print(f'node3 : {node3}')

sess = tf.Session()

# print(f'sess.run(node1,node2) : {sess.run(node1,node2)}')
print(f'sess.run(node1,node2,node3) : {sess.run([node1,node2,node3])}')

res = sess.run(node4)
print(res)

mul = tf.multiply(node1,node2)
print(f'3 * 4 : {sess.run(mul)}')

sub = tf.subtract(node1,node2)
print(f'3 * 4 : {sess.run(sub)}')