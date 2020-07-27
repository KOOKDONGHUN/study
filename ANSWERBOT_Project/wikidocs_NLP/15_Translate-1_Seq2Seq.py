import pandas as pd
import urllib3 # load internet resource
import zipfile # unpacking .zip file
import shutil # use to copy, remove file
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np


http = urllib3.PoolManager()
url = 'http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd() # get absolute path on working directory

zipfilename = os.path.join(path, filename)

""" # 인터넷에 있는 압축파일을 현재 나의 디렉토리 밑에 복사함 
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref :
    zip_ref.extractall(path) # path에 존재하는 압축파일 압축풀기 """

lines = pd.read_csv('fra.txt', names=['src', 'tar','_'], sep='\t')
print(len(lines)) # 177210
print(lines.head(10))

# 데이터 양이 많기때문에 6만개만 사용
lines = lines.loc[:, 'src':'tar']
lines = lines.iloc[0:60000]
print(lines.sample(10))
"""                            src                                          tar
    11993          Life's not easy.                     La vie n'est pas facile.
    29729      I know what you did.                   Je sais ce que tu as fait.
    43462    Let sleeping dogs lie.  Il ne faut pas réveiller le chien qui dort.
    25411       Keep off the grass.                 Ne pas marcher dans l'herbe.
    43174    It could be dangerous.                Cela pourrait être dangereux.
    3784              I want these.                           Je veux celles-là.
    55517  I swim almost every day.              Je nage presque tous les jours.
    652                  I relaxed.                         Je me suis détendue.
    46609   Can you close the door?                    Peux-tu fermer la porte ?
    54281  He's going to get fired.                        Il va se faire virer.    """


lines.tar = lines.tar.apply(lambda x : '\t ' + x + ' \n') # 이렇게 하면 반복문을 돌면서 적용할 필요없이 한번에 적용된까 코드상으로는 보기 좋네
print(lines.sample(10))

# target 값에 대한 시작 심볼과 종료 심볼이 추가됐다. 단어 단위가 아닌 글자단위의 토큰화 진행 (글자 집합 생성)
src_vocab = set() # 중복에 대해 제거하기 위해서?? set??
for line in lines.src:
    for char in line :
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print(src_vocab_size)
print(tar_vocab_size)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab[:45])
print(tar_vocab[:45])

# for i in src_vocab:
#     try :
#         tar_vocab.remove(i)
#     except :
#         pass
# print(tar_vocab)

# 인코딩하는 과정중 하나 인덱스화 -> 컴퓨터가 이해 할 수 있도록 숫자로 변환하기 위한 작업
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
print(src_to_index.keys())
print(src_to_index.values())

tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(tar_to_index)

encoder_input = []
for line in lines.src :
    temp_x = []
    for w in line :
        temp_x.append(src_to_index[w])
    encoder_input.append(temp_x)
print(f'encoder_input : {encoder_input[:5]}')

decoder_input = []
for line in lines.tar :
    temp_x = []
    for w in line :
        temp_x.append(tar_to_index[w])
    decoder_input.append(temp_x)
print(f'decoder_input : {decoder_input[:5]}')

# 디코더는 왜 인풋값과 타겟값을 둘다 디코딩 해주는거지?? -> 디코더의 예측값과 비교하기 위한 실제값이 필요하다. 그러면 인풋에서도 마지막 심볼 제거 안함?
decoder_target = []
for line in lines.tar:
    t = 0
    temp_x = []
    for w in line :
        if t > 0: # \t 거르기 그럼 마지막은?
            temp_x.append(tar_to_index[w])
        t += 1
    decoder_target.append(temp_x)
print(decoder_target[:5])

# 패딩 작업 전 가장긴 문장의 길이 찾기
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print(max_src_len)
print(max_tar_len)

# 각각의 데이터에 대해 패딩 처리
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

# 디코더의 경우 인풋과 타겟이 있는 이유 -> 이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로 바로 넣어주지 않고,
# 이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 하는 방법을 사용하기 때문이다. 이렇게 하는 이유도 잘 모르겠다....

# model
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # 이 부분이 컨텍스트 벡터이다.

# seq2seq는 히든스테이스와 셀스테이트를 디코더로 넘겨줌 ...
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# 디코더의 첫 상태를 인코더의 은닉 상태, 셀 상태로 한다?? 인코더와 디코더의 연결을 말하는 건가
# 디코더는 인코더의 마지막 은닉 상태를 초기 은닉 상태로 사용한다.
decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax') # 근데 함수 형이면 이친구 뒤에
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile, fit
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=50, validation_split=0.2)

