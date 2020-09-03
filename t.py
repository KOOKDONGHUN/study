from keras.layers import LSTM, Embedding
from keras.models import Sequential, Model, Input


input1 = Input(shape=(None,))
embedding = Embedding(6406, 100, input_length=None)(input1) # (None, 79, 100) 
lstm1 = LSTM(128,
            dropout=0.1,
            recurrent_dropout=0.5)(embedding)

model = Model([input1], [lstm1])
model.summary()