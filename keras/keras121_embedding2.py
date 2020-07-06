from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', # '최고에요' -> '참 최고에요'
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
# token.fit_on_texts([docs]) 리스트 자체에서 토큰화
token.fit_on_texts(docs) # 리스트네의 요소들에 대해 토큰화
print(token.word_index)

# 많이 나온 순서 '참'을 리스트의 요소에 추가하면 '너무' 와 '참'의 순서가 바뀐다

x = token.texts_to_sequences(docs)
print(x)

'''{'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
 '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19,
  '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23}
  
[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]'''

# 원핫 인코딩?
from keras.preprocessing.sequence import pad_sequences 
# default 0과 pre
# pad_x = pad_sequences(x, padding='pre') # 0이 앞으로
# pad_x = pad_sequences(x, padding='post') # 0이 뒤로
pad_x = pad_sequences(x, padding='post', value=0) # 0이 뒤로

print(pad_x)