from konlpy.tag import Kkma
from konlpy.utils import pprint
kkma = Kkma()
print(kkma.sentences(u'JPype 설치 너무 까다롭습니다. 몇 시간을 날렸어요.'))

print(kkma.nouns(u'JPype 설치 너무 까다롭습니다. 몇 시간을 날렸어요.'))

print(kkma.pos(u'JPype 설치 너무 까다롭습니다. 몇 시간을 날렸어요.'))