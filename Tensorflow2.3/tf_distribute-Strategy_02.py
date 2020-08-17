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
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Maxpooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Maxpooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')]
        )
        
        return model

checkpoint_dir = f'./Model/{__file__.split(".")[0]}_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction = tf.keras.losses.Reduction.None
    )
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)

        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')