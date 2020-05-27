# 두번째 답
import numpy as np
from keras.utils import np_utils

y = np.array(['트와이스','소녀시대','레드벨벳','우주소녀','트와이스','레드벨벳','블랙핑크','오마이걸'])
print("y : {y}")
y = np_utils.to_categorical(y)
print("y : {y}")