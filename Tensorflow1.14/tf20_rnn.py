import tensorflow as tf
import numpy as np

### 1. data
# 수치화, 와꾸 맞추기
# hihello -> 수치화
idx2_char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1,1)
# _data = np.array([['h'], ['i'], ['h'], ['e'], ['l'], ['l'], ['o']])
print(_data.shape)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
print('-'*40,'\n',_data.shape)
print(_data)
print(type(_data))

x_data = _data[:6, ] # hihell
y_data = _data[1:, ] #  ihello
print('-'*40)
print(x_data)
print(y_data)

y_data = np.argmax(y_data, axis=1)
print('-'*40, 'np.argmax(y_data, axis=1)')
print(y_data)
print(y_data.shape)

# reshape 왜 함????
x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print('-'*40)
print(x_data.shape)
print(y_data.shape)

sequence_length = x_data.shape[1]
input_dim = x_data.shape[2]

# X = tf.placeholder(tf.float32, shape=[None, sequence_length, input_dim]) # 같은말
# X = tf.placeholder(tf.float32, (None, sequence_length, input_dim)) # 같은말
X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.placeholder(tf.float32, shape=[None, sequence_length]) # 같은말
# Y = tf.placeholder(tf.float32, (None, sequence_length)) # 같은말
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print('-'*40)
print(X)
print(Y)
### 1. data

### 2. model  # LSTM -> 두번 연산하는것을 tf에서는 명시를 해준다 ???
output_node = 5
batch_size = 1 # 전체 행??
lr = 0.01

### model.add(LSTM(output_node###, input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output_node) # _lstm  ## 여기가 첫번째 연산
cell = tf.keras.layers.LSTMCell(output_node) # _lstm  ## 여기가 첫번째 연산
h, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # == model.add(LSTM()) ## 여기가 두번째 연산?
print(h) # Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32)

w = tf.ones([batch_size, sequence_length]) # == Y.shape // 1로 해도 값의 변화가 크지 않다 
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=h, targets=Y, weights=w) #LSTM에서는 loss를 무조건 이거를 써야한다?  mse를 풀어썻다? // loss = h - y 가 기본
cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# prediction = tf.argmax(h, axis=2)
prediction = tf.arg_max(h, dimension=2)
### 2. model

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(f'epochs : {i}\t loss : {loss}\t prediction : {result}\t y_true : {y_data}')

    res_str = [idx2_char[c] for c in np.squeeze(result)]
    # print(res_str)
    print('Prediction str : ',''.join(res_str))