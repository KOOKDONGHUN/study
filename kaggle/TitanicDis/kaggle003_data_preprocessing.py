import pandas as pd # 판다스데이터 프레임으로 구조를 만들때는 인자로 딕의 형태를 받음 -> pd.DataFrame(dic)
import numpy as np

# 1. 데이터 불러오기
train_data = pd.read_csv('c:/titanic/train.csv') 
test_data = pd.read_csv('c:/titanic/test.csv')

'''
- PassengerId : 승객 번호
- Survived : 생존여부(1: 생존, 0 : 사망)
- Pclass : 승선권 클래스(1 : 1st, 2 : 2nd ,3 : 3rd)
- Name : 승객 이름
- Sex : 승객 성별
- Age : 승객 나이 
- SibSp : 동반한 형제자매, 배우자 수
- Patch : 동반한 부모, 자식 수
- Ticket : 티켓의 고유 넘버
- Fare 티켓의 요금
- Cabin : 객실 번호
- Embarked : 승선한 항구명(C : Cherbourg, Q : Queenstown, S : Southampton)'''

'''데이터 구조? 눈으로 직접 보기 위한 프린트'''
# print(f"train_data.head() : {train_data.head()}")
feature = train_data.columns  # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# print(f"feature : {feature}")

# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# temp = feature[4] # Sex
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
# pie_chart(temp) # test
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


'''인덱스가 흔하지 않은 title인 것을 단순히 Other로 바꾼다 -> 왜? 바꾸지 말고 차라리 버리면? '''
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

'''train_data[,,] -> 이런식으로 []안에 ,로 여러가지 컬럼들을 입력해서 데이터를 원하는 컬럼끼리만 볼수 있는듯 하다. '''

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
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['embarked'] = dataset['Embarked'].astype(str)
print(f"transform -> Embarked.value_counts :\n{train_data.Embarked.value_counts(dropna=False)}")

'''나이 평균 구하기 Age의 결측치도 지금은 평균으로 채워주지만 다른 방법을 생각해보자'''
print("train_data.Age.value_counts(dropna=False) : \n",train_data.Age.value_counts(dropna=False)) # 177명인데 이걸 평균으로만 채우면 예측률에 영향이 있을듯?
avg_Age = round(train_and_test[0]['Age'].mean(),2) # 29.69
print("avg_Age = ",avg_Age)

for dataset in train_and_test:
    dataset['Age'].fillna(train_and_test[0]['Age'].mean(),inplace=True)
    # dataset['Age'] = dataset['Age'].fillna(train_and_test[0]['Age'].mean(),inplace=True) # ㅋㅋ 이것 떄문에 안된건데 뭐가 문제 일까 
    # print("1")
    dataset['Age'] = dataset['Age'].astype(int) # -> 뭐야 왜 에러야 TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'
    # print("2")
    # 이거 인헤도 어차피 int형 일거 같은데 일단 생략 해봄 추후에 이것 떄문에 에러가 난다면 수정해야함!!

    train_data['AgeBand'] = pd.cut(train_data['Age'], 5) # 어차피 트레인 데이터만 할거면 반복문 안에다 쓴이유를 찾아보고 이 의미를 알아보자 

# 뭐지 NaN만 남기고 다 사라짐 -> 이렇게된 문제점은 찾았지만 왜 없어졌는지는 모르겠다
print("train_data.Age.value_counts(dropna=False) : \n",train_data.Age.value_counts(dropna=False))


print (train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean(),"\n") # Survivied ratio about Age Band
'''      AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.344762
2   (32.0, 48.0]  0.403226
3   (48.0, 64.0]  0.434783
4   (64.0, 80.0]  0.090909'''


''' Age의 구간을 정하는 이유? '''
for dataset in train_and_test:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
'''이 블로그의 글쓴이는 Age값으로 numeric이 아닌 string의 형식으로 넣어 주었다는데 숫자에 대한 경향성을 가지고 싶지 않다고 함 뭔 소린지 모르겠다...
     -> 이유없음 없어도 되는 부분인듯 하다'''

print ("",train_data[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
'''
    Pclass       Fare
0       1  84.154687
1       2  20.662183
2       3  13.675550'''

print("")

print(test_data[test_data["Fare"].isnull()]["Pclass"]) # .isnull() -> 결측치 찾기 
'''
152    3
Name: Pclass, dtype: int64'''

for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

'''형제, 자매, 배우자, 부모님, 자녀의 수가 많을 수록 생존한경우가 많다는 것을 위에 그린 그래프를 보면 알수 있다
   두개의 칼럼을 하나의 칼럼으로 만들어 준다. '''
for dataset in train_and_test:
    dataset['Family'] = dataset['Parch'] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)

'''특성 추출 및 나머지 전처리 '''
features_drop = ['Name','Ticket','Cabin','SibSp','Parch']
train_data = train_data.drop(features_drop,axis=1)
test_data = test_data.drop(features_drop,axis=1)
train_data = train_data.drop(['PassengerId','AgeBand','FareBccand'],axis=1)

print("train_data : \n",train_data.head())
print("train_data.shape : \n",train_data.shape)

print("test_data : \n",test_data.head())
print("test_data.shape : \n",test_data.shape)