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

print("type of users : ", type(users)) # type of users :  <class 'list'>
print("type of users[0] : ", type(users[0])) # type of users[0] :  <class 'dict'>

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
'''---- 데이터 영역 ----'''


'''---- 이 과정의 목적은 각 사용자간의 연결된 수(네트워크 수)를 구하기 위함 ----'''
# 사용자별로 친구목록을 저장하기위한 비어있는 리스트를 생성
friendships = {user["id"] : [] for user in users}
print("friendships : ",friendships) # {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

# friendship_pairs의 쌍을 이용하여 친구목록 리스트 작성
for  i,j in friendship_paris:# 반복문 첫 수행 결과   ↓ ↓ ↓ ↓          ↓ ↓ ↓ ↓ ↓                ↓ ↓ ↓ ↓
    friendships[i].append(j) # {0: [1], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    friendships[j].append(i) # {0: [1], 1: [0], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

# 최종 결과 
print("friendships : ",friendships) # friendships :  {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4],
                                                   #  4: [3, 5], 5: [4, 6, 7], 6: [5, 8], 7: [5, 8],
                                                   #  8: [6, 7, 9], 9: [8]}
'''----- 이 과정의 목적은 각 사용자간의 연결된 수(네트워크 수)를 구하기 위함 ----'''


'''---- 네트워크의 평균 ----'''
def number_of_friends(user):
    ''' user를 넣었을 때 그 사람의 친구는 몇명인가? 대답해주는 함수  '''
    user_id = user["id"] # 함수를 호출 할 때 반복문을 사용하여 딕이 하나씩 user라는 파라미터로 전달되면 딕 안에서 키값이 "id"인 것을 변수 user_id에 저장한다.
    friend_ids = friendships[user_id] # 위에서 저장된 id 값 예를들면 0을 인덱스로 사용하여 friendships[0] -> [1, 2] 값을 friend_ids 변수에 저장
    return len(friend_ids) # 처음 호출시 [1, 2] 반환


total_conn = sum(number_of_friends(user) for user in users) # 24
''' 한줄 반복문을 일반 반복문으로 바꾸면

     for i in len(users) :         |  또는   for user in users :
        temp = users[i]            |             number_of_friends(user)
        number_of_friends(temp)    |                                        '''

num_of_users = len(users) # 10
avg_conn = total_conn / num_of_users # 24/ 10 == 2.4
'''---- 네트워크의 평균 ----'''


'''---- 네트워크의 수가 가장 많은 사람 -> 최강 인싸 찾기 ----'''
id_is_num_of_friend = [(user["id"],number_of_friends(user)) for user in users] 
''' 한줄 반복문을 일반 반복문으로 바꾸면

    id_is_num_of_friend = []        
    for user in users :
        user_id = user["id"]
        user_friend_num = number_of_friends(user)
        id_is_num_of_friend.append((user_id,user_friend_num))

    print("id_is_num_of_friend : ",id_is_num_of_friend) 
    프린트 결과 표시 -> id_is_num_of_friend :  [(0, 2), (1, 3), (2, 3), (3, 3), (4, 2), (5, 3), (6, 2), (7, 2), (8, 3), (9, 1)] '''
    
# 최강 인싸를 찾기 위한 정렬
id_is_num_of_friend.sort( key = lambda id_and_friends : id_and_friends[1], # number_of_friends(user)를 기준으로 정렬하겠다.
                          reverse = True ) # 내림차순으로 하겠다.
print("id_is_num_of_friend : ",id_is_num_of_friend) # id_is_num_of_friend :  [(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]
'''---- 네트워크의 수가 가장 많은 사람 -> 최강 인싸 찾기 ----'''


''' ---- 친구의 친구 추천 프로그램 ----'''
# 친구의 친구를 추천 해주는 함수
def fri_of_a_fri_recommend(users = {}):
    return [foaf_id for friend_id in friendships[users["id"]]
                        for foaf_id in friendships[friend_id]]

# friendships :  {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4],
                                                   #  4: [3, 5], 5: [4, 6, 7], 6: [5, 8], 7: [5, 8],
                                                   #  8: [6, 7, 9], 9: [8]}

print("users[0] : ",users[0]) # {'id': 0, 'name': 'Hero'}
recommend_for_user0 = fri_of_a_fri_recommend(users[0])
print("recommend_for_user0 : ", recommend_for_user0) # recommend_for_user0 :  [0, 2, 3, 0, 1, 3]
''' 문제점 발생 1. 자기 자신과도 친구임으로 결과가 나옴
                2. 이미 친구인 1,2를 추천해야함
                3. 3을 중복으로 추천해줌 '''
'''
temp = []
for friend_id in friendships[users[0]["id"]] :  # ex) users[0] == {'id': 0, 'name': 'Hero'}
                                                #     users[0]["id"] == 0
                                                #     friend_id <- friendships[0] == [1,2]
    print("friend_id : ", friend_id)
    for foaf_id in friendships[friend_id] :  # foaf_id <- friendships[1] == [0,1,3]
        temp.append(foaf_id)
        print("foaf_id : ", foaf_id)
        print("temp : ",temp)
print(recommend_for_user0, " == ", temp) '''
''' ---- 친구의 친구 추천 프로그램 ----'''


''' ---- 친구의 친구 추천 프로그램 (자기자신과 중복되는 친구를 제외) ----'''
from collections import Counter

def friends_of_friends(user):
    user_id = user["id"]
    return Counter( foaf_id for friend_id in friendships[user_id]
                                for foaf_id in friendships[friend_id]
                                    if (foaf_id != user_id) and (foaf_id not in friendships[user_id]) ) 
                                                                # x not in 리스트 : 리스트에 x가 없으면 True
mutualfriends_count = friends_of_friends(users[0])
print("mutualfriends_count : ",mutualfriends_count) # mutualfriends_count :  Counter({3: 2})
                                                    # 중복과 자기 자신을 제외하면 3번 친구와 아는 사람이 2명이 겹친다.
''' ---- 친구의 친구 추천 프로그램 (자기자신과 중복되는 친구를 제외) ----'''