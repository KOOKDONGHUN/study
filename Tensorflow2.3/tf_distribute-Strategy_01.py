import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar() # 다운로드시 프로그래스바 노출을 하지 않기 위함 인듯?

import os

# data load, split
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# define Strategy
strategy = tf.distribute.MirroredStrategy()