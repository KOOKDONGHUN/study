import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

'''
Survivied는 생존 여부(0은 사망, 1은 생존; train 데이터에서만 제공),
Pclass는 사회경제적 지위(1에 가까울 수록 높음),
SipSp는 배우자나 형제 자매 명 수의 총 합,
Parch는 부모 자식 명 수의 총 합을 나타낸다.

'''

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())

'''
트레인 셋과 테스트 셋의 컬럼의 갯수가 다른 이유?
테스트에는 생존 여부가 없음 있어야 테스트한게 맞는지 틀린지 알거아닌가?

각 컬럼들과 생존간의 연관성을 그래프로 그리기
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots