import tensorflow as tf

import numpy as np
import os

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# (batch_size)

x_train = x_train[..., None]
x_test = x_test[..., None]

x_train = x_train / np.float32(255)

strategy = tf.distribute.MirroredStrategy()

print(f'장치의 수 : {strategy.num_replicas_in_sync}')

BUFFER_SIZE = len(x_train)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

with strategy.scope():
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_datasets)

    def create_model():
        model = tf.keras.Sequential()