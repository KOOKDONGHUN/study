import pandas as pd # 판다스데이터 프레임으로 구조를 만들때는 인자로 딕의 형태를 받음 -> pd.DataFrame(dic)
import numpy as np



# 1. 데이터 불러오기
train_data = pd.read_csv('./data/csv/train.csv')
test_data = pd.read_csv('./data/csv/test.csv')

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

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
sns.set() # setting seaborn default for plots 여기서는 사용하지 않았음 

# print(f"train_data :\n{train_data}")

# categorical feature의 분포(연관성?)를 보기 위해서 pie chart
def pie_chart(feature):

    # train_data 의 칼럼명이 feature인 데이터 종류와 개수? 
    feature_ratio = train_data[feature].value_counts(sort=False)
    print(f"\nfeature_ratio : \n{feature_ratio}") #feature_ratio : female    314    \n male      577    \n Name: Sex, dtype: int64
    print(f"\nfeature_ratio.len : \n{len(feature_ratio)}") #feature_ratio.len : 3 -> 이거는 들어오는 컬럼에 따라 달라지는데 
                                                           #                        컬럼의 세부분류가 몇개 인가? 정도가 적당한 설명 같다 
    print(f"\nfeature_ratio.shape : \n{feature_ratio.shape}") #feature_ratio.shape : (3,) -> 마찬가지 
    
    # len과 같은말 
    feature_size = feature_ratio.size
    print(f"\nfeature_size : \n{feature_size}") # 3

    # train_data 의 칼럼명이 feature인 데이터의 종류(인덱스 이름)
    feature_index = feature_ratio.index
    print(f"\nfeature_index : \n{feature_index}") # Index(['female', 'male'], dtype='object')

    # 선착장 또는 좌석의 등급(퍼스트 이코노미 같은)을 기준으로 생존한 인원수 
    survived = train_data[train_data['Survived'] == 1][feature].value_counts()
    print(f"\nsurvived : \n{survived}")

    # 선착장 또는 좌석의 등급(퍼스트 이코노미 같은)을 기준으로 생존한 인원수 
    dead = train_data[train_data['Survived'] == 0][feature].value_counts()
    print(f"\ndead : \n{dead}")


    plt.plot(aspect='auto')# 데이터를 자동으로 사각형에 채우겠다? 그래프의 막대 또는 원에서 높이와 넓이를 적당한 사이즈로? 자동으로 맞추겠다?
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')# autopct='%1.1f%%' -> 파이 조각의 전체 대비 백분율을 소수점 1자리까지 표기한다
    plt.title(feature + '\'s ratio in total')

    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()

# 차트를 보기위한 함수 실행 
# pie_chart(temp)

pie_chart('Pclass') #-> 왜 안될까?     # print(f"feature_ratio[0] : {feature_ratio[0]}") # 577 -> 이 부분에서 에러가남 -> []에 들어가는 부분이 인덱스의 숫자가 아니고 인덱스의 이름으로 들어간다

pie_chart('Embarked')

def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead]) 
    df.index = ['Survived','Dead']
    df.plot( kind = 'bar', stacked = True, figsize = (10, 5))
    plt.show()

bar_chart("SibSp") # - SibSp : 동반한 형제자매, 배우자 수
bar_chart("Parch") # - Patch : 동반한 부모, 자식 수