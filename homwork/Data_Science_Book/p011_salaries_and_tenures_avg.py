'''---- 데이터 영역 ----'''
users = [
    {"id":0,"name":"Hero"},
    {"id":1,"name":"Dunn"},
    {"id":2,"name":"Sue"},
    {"id":3,"name":"Chi"},
    {"id":4,"name":"Thor"},
    {"id":5,"name":"Clive"},
    {"id":6,"name":"Hiicks"},
    {"id":7,"name":"Devin"},
    {"id":8,"name":"kate"},
    {"id":9,"name":"Kein"},
]

friendship_paris = [ (0,1),(0,2),
                     (1,2),(1,3),
                     (2,3),(3,4),
                     (4,5),(5,6),
                     (5,7),(6,8),
                     (7,8),(8,9)
]

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoQDL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"), 
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"),(6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data"),
]

salaries_and_tenures = [(83000,8.7),(88000,8.1),
                        (48000,0.7),(76000,6),
                        (69000,6.5),(76000,7.5),
                        (60000,2.5),(83000,10),
                        (48000,1.9),(63000,4.2)]
                        
'''---- 데이터 영역 ----'''


'''---- 특정 관심사를 가지고 있는 모든 사람의 id를 반환하는 함수 ----'''
def data_scientists_who_like(target_interest):
    return [user_id for user_id, user_interest in interests
                        if user_interest == target_interest]
# 문제점 호출할 때마다 관심사 데이터를 매번 처음부터 끝까지 훑어야 한다는 단점이 있다.
# 사용자 수가 많고 관심사가 많은 데이터라면 각 관심사로 사용자 인덱스를 만드는 것이 좋은 방법이다. ???
from collections import defaultdict, Counter
 
 # 키가 관심사, 값이 사용자 id
user_ids_by_interest = defaultdict(list)

# 키가 사용자, 값이 관심사
interests_by_user_ids = defaultdict(list)

for user_id, interest in interests :
    # 키가 관심사, 값이 사용자 id
    user_ids_by_interest[interest].append(user_id)

    # 키가 사용자, 값이 관심사
    interests_by_user_ids[user_id].append(interest) 
print("interests_by_user_ids : ",interests_by_user_ids)
print("user_ids_by_interest : ",user_ids_by_interest)
'''---- 특정 관심사를 가지고 있는 모든 사람의 id를 반환하는 함수 ----'''

def most_common_interests_with(user) :
    return Counter(
        interested_user_id for interest in interests_by_user_ids[user["id"]]
                                for interested_user_id in user_ids_by_interest[interest]
                                    if interested_user_id != user["id"]
    )

#키는 근속연수, 값은 해당 근속 연수에 대한 연봉 목록
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures :
    salary_by_tenure[tenure].append(salary)
print("tenure :",tenure) # tenure : 4.2 왜 같은 변수명으로 밑에서도 쓰냐 사람 햇갈리게

# 키는 근속연수, 값은 해당 근속 연수의 평균 연봉
average_salary_by_tenure = {
    tenure : sum(salaries) /  len(salaries) for tenure ,salaries in salary_by_tenure.items()
}
print("average_salary_by_tenure : ",average_salary_by_tenure)
''' average_salary_by_tenure :  {8.7: 83000.0, 8.1: 88000.0, 0.7: 48000.0,
                                 6: 76000.0, 6.5: 69000.0, 7.5: 76000.0,
                                 2.5: 60000.0, 10: 83000.0, 1.9: 48000.0,
                                 4.2: 63000.0} '''

# 경력 구간 나누기
def tenure_bucket(tenure):
    if tenure < 2 :
        return "less than two"
    elif tenure < 5 :
        return "between two and five"
    else :
        return "more than five"

# 각 연봉을 해당구간에 대응 시켜보기.
salary_by_tenure_bucket = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# 키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 평균 연봉
average_salary_by_bucket = {
    tenure_bucket : sum(salaries) / len(salaries) for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}
print("average_salary_by_bucket : ",average_salary_by_bucket)
''' average_salary_by_bucket :  {'more than five': 79166.66666666667, 'less than two': 48000.0, 'between two and five': 61500.0} '''

