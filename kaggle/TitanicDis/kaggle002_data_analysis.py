import pandas as pd # 판다스데이터 프레임으로 구조를 만들때는 인자로 딕의 형태를 받음 -> pd.DataFrame(dic)
import numpy as np



# 1. 데이터 불러오기
train_data = pd.read_csv('c:/titanic/train.csv') 
test_data = pd.read_csv('c:/titanic/test.csv')



'''데이터 구조? 눈으로 직접 보기 위한 프린트'''

# print(f"train_data.head() : {train_data.head()}")
feature = train_data.columns  # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# print(f"feature : {feature}")

# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
temp = feature[4] # Sex
# print(f"train_data : {train_data[temp]}\n")

# print(f"feature[0] : {feature[0]}")
# print(f"feature.type : {type(feature)}")

# print('train_data data shape: ', train_data.shape)
# print('test data shape: ', test.shape)
# print('----------[train_data infomation]----------')
# print(train_data.info()) # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# print('----------[test infomation]----------')
# print(test.info()) # PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

'''데이터 구조? 눈으로 직접 보기 위한 프린트'''



'''
Survivied는 생존 여부(0은 사망, 1은 생존; train 데이터에서만 제공),
Pclass는 사회경제적 지위(1에 가까울 수록 높음),
SipSp는 배우자나 형제 자매 명 수의 총 합,
Parch는 부모 자식 명 수의 총 합을 나타낸다.

'''



'''
트레인 셋과 테스트 셋의 컬럼의 갯수가 다른 이유?
테스트에는 생존 여부가 없음 있어야 테스트한게 맞는지 틀린지 알거아닌가?

각 컬럼들과 생존간의 연관성을 그래프로 그리기
'''

# 그래프 그리기 준비
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots

# categorical feature의 분포(연관성?)를 보기 위해서 pie chart
def pie_chart(feature):

    # train_data 의 칼럼명이 feature인 데이터 종류의 개수? 말로 표현하기 어렵네 
    feature_ratio = train_data[feature].value_counts(sort=False)
    print(f"feature_ratio : {feature_ratio}") #feature_ratio : female    314    \n male      577    \n Name: Sex, dtype: int64
    print(f"feature_ratio.len : {len(feature_ratio)}") #feature_ratio.len : 2
    print(f"feature_ratio[0] : {feature_ratio[0]}") # 577
    print(f"feature_ratio[1] : {feature_ratio[1]}") # 314

    # len과 같은말 
    feature_size = feature_ratio.size
    print(f"feature_size : {feature_size}") # 2

    # train_data 의 칼럼명이 feature인 데이터의 종류(인덱스 이름)
    feature_index = feature_ratio.index
    print(f"feature_index : {feature_index}") # Index(['female', 'male'], dtype='object')

    #
    survived = train_data[train_data['Survived'] == 1][feature].value_counts()
    dead = train_data[train_data['Survived'] == 0][feature].value_counts()
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    # plt.show()
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    
    # plt.show()

# 차트를 보기위한 함수 실행 
pie_chart(temp)
