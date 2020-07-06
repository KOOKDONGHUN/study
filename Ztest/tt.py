import tensorflow as tf

vocab = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']

indices = tf.range(len(vocab),dtype=tf.int64)
print(indices)

table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
print(table_init)

num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
print(table)