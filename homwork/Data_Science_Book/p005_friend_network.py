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

# 이 과정의 목적은 각 사용자간의 연결된 수(네트워크 수)를 구하기 위함