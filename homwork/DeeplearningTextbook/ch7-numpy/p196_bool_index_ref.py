import numpy as np

t = True
f = False

arr = np.array([2, 4, 6, 7])

print(arr[np.array([t,t,t,f])]) # [2 4 6]

'''이런식으로 인덱스 값에 불값을 넣어서 슬라이싱? 도 가능'''
arr = np.array([2,4,6,7])
print(arr[arr % 3 == 1]) # 4 7

한글 = "a"
print(한글)

arr= np.array([2,3,4,5,6,7])
print(arr % 2 == 0)
print(arr[arr % 2 == 0])