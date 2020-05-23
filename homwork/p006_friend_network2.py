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
        number_of_friends(temp)   |                                        '''

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
