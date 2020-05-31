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

    # 성별을 기준으로 생존한 인원수 
    survived = train_data[train_data['Survived'] == 1][feature].value_counts()
    print(f"survived : {survived}")

    # 성별을 기준으로 사망한 인원수
    dead = train_data[train_data['Survived'] == 0][feature].value_counts()
    print(f"dead : {dead}")


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

# pie_chart('Pclass') -> 왜 안될까?

pie_chart('Embarked')

def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead]) 
    df.index = ['Survived','Dead']
    df.plot( kind = 'bar', stacked = True, figsize = (10, 5))
    # plt.show()

bar_chart("SibSp")
bar_chart("Parch")
train_and_test = [train_data,test_data]

for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.') 
'''-> '([A-Za-z]+)\.' 정규 표현식인데 공백으로 시작하고 ,로 끝나는 문자열을 추출할 사용하는 표현 방식이다'''

print(train_data.head(5))

'''추출한 Title을 가진 사람이 몇 명이 존재 하는지 성별과 함께 나타내보기'''
title_sex = pd.crosstab(train_data['Title'],train_data['Sex'])
print(f"title_sex : \n{title_sex}")
print(f"title_sex.shape : \n{title_sex.shape}") # 17,2 title이 인덱스 처럼 들어가네 


'''인덱스가 흔하지 않은 title인 것을 단순히 Other로 바꾼다 -> 왜?'''
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess',
                                                 'Don','Dona', 'Dr',
                                                 'Jonkheer','Lady','Major',
                                                 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train_data[['Title','Survived']].groupby(['Title'], as_index=False).mean()
print(f"train_data[['Title','Survived']].groupby(['Title'], as_index=False).mean() : \n{train_data[['Title','Survived']].groupby(['Title'], as_index=False).mean()}")

'''추출한 title데이터를 학습하기 알맞게 StringData로 변형해준다. -> 왜? 학습하기 알맞은 이유? '''

'''train_data[,,] -> 이런식으로 []안에 ,로 여러가지 컬럼들을 입력해서 데이터를 원하는 컬럼끼리만 볼수 있는듯 하다. 
   프린트 찍어서 확인해보자!'''

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)


'''성별은 male과 female로 나뉘어 있으므로 stringdata로 만 변형해주면 된다
   그럼 원래는 타입이 뭐가 나오는가 위에도 마찬가지고 타입을 변형해주는 이유
print(f"type(dataset['Sex']) : \n{type(dataset['Sex'])}") # <class 'pandas.core.series.Series'>
print(f"type(dataset['Sex'][0]) : \n{type(dataset['Sex'][0])}") # <class 'str'>'''

for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)

'''print(f"type(dataset['Sex'][0]) : \n{type(dataset['Sex'][0])}") # <class 'str'>
    print(f"type(dataset['Sex']) : \n{type(dataset['Sex'])}") # <class 'pandas.core.series.Series'>
    뭘 바꿨다는 거야... 그대로 나오는데 ... 왜 바꾸는 걸까'''

train_data.Embarked.value_counts(dropna=False)
print(f"train_data.Embarked.value_counts(dropna=False) :\n{train_data.Embarked.value_counts(dropna=False)}")
'''블로그 오타 있음...  Embared -> Embarked  value_count -> value_counts 
   중요한건 오타가 문제가 아니고 주어진 데이터에서 Nan값이 있다는거 결측치를 의미하는거 같다
   블로그에서는 결측치를 s로 넣어줬는데 시간이 된다면 다른 걸 넣어서 테스트 해보자!!'''
for  dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['embarked'] = dataset['Embarked'].astype(str)
print(f"transform -> Embarked.value_counts :\n{train_data.Embarked.value_counts(dropna=False)}")

'''나이 평균 구하기 Age의 결측치도 지금은 평균으로 채워주지만 다른 방법을 생각해보자'''
def avg(col):
    avg_Age = 0
    for i in train_and_test[col]:
        avg_Age += i
    res = avg_Age/len(train_and_test[col])
    return res
print(avg())
# for  dataset in train_and_test:
#     dataset['Age'] = dataset['Age'].fillna('S')
#     dataset['Age'] = dataset['Age'].astype(str)