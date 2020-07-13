import tensorflow as tf
import numpy as np

def split_x(seq, size=3):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

def split_x2(seq, size=3):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i+1:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# print(datasets.shape)

# RNN model 
''' start = 0
end = 4

x_data = np.array(list())
y_data = np.array(list())

for i in range(len(datasets)):
    try :
        x_data = datasets[start:end]
        y_data = datasets[start+1:end+1]

        start += 1
        end += 1
    except:
        break

print(x_data)
print(x_data.shape)

print(y_data)
print(y_data.shape) '''

''' x_data = split_x(datasets,4)[:-1]
print(x_data)

y_data = split_x2(datasets,5)
print(y_data)

print(x_data.shape)
print(y_data.shape) '''

_data = split_x(datasets, 4)
print(_data)
print(_data.shape)

x_data = _data[:, :-1]
y_data = _data[:, 1:]

print(x_data)
print(x_data.shape)

print(y_data)
print(y_data.shape)

x_data = x_data.reshape(x_data.shape[0],x_data.shape[1],1)

print(x_data.shape)
print(y_data.shape)

sequence_length = x_data.shape[1]
input_dim = x_data.shape[2]

print(sequence_length)
print(input_dim)

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X)
print(Y)
''' 
# model
output_node = 1
batch_size = x_data.shape[0] # 전체 행??
lr = 0.01

cell = tf.keras.layers.LSTMCell(output_node) # _lstm  ## 여기가 첫번째 연산
h, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # == model.add(LSTM()) ## 여기가 두번째 연산?
print(h) # Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32)


w = tf.ones([batch_size, sequence_length]) # == Y.shape // 1로 해도 값의 변화가 크지 않다 
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=h, targets=Y, weights=w) #LSTM에서는 loss를 무조건 이거를 써야한다?  mse를 풀어썻다? // loss = h - y 가 기본
cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)

prediction = tf.math.argmax(h, axis=2)


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(f'epochs : {i}\t loss : {loss}\t prediction : {result}\t y_true : {y_data}') '''