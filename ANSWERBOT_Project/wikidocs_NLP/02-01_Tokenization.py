from konlpy.tag import Okt
okt = Okt()
text = "열심히 고딩한 당신, 연휴에는 여행을 가봐요. 안녕하세요"
print(okt.morphs(text))

import kss
print(kss.split_sentences(text))