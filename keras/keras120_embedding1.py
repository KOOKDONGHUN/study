from keras.preprocessing.text import Tokenizer # 토큰이 무엇인가

text = "나는 맛있는 밥을 먹었다."

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # 나는이 밥을의 3배의 가치? 라벨인코더를 사용하지 않았을떄?ㅡ왜 3배?

x = token.texts_to_sequences([text])
print(x)

from keras.utils import to_categorical

word_size = len(token.word_index) +1 # 1을 더하는 이유?
x = to_categorical(x, num_classes=word_size)
print(x)

# 압축하는 방법의 하나가 임베딩이다. 시계열에서 주로 사용함