# 자료형
# 1. 리스트
a = [1,2,3,4,5]
b = [1,2,3,'a','b']
print(a)
print(b)
print(a[0] + a[3])
# print(b[0] + b[3]) # TypeError: unsupported operand type(s) for +: 'int' and 'str'

print(type(a)) # list
print(a[-2]) # 4
print(a[1:3]) # 2,3

a = [1,2,3,['a','b','c']]
print(a[1]) # 2
print(a[-1]) # ['a','b','c']
print(a[-1][1]) # b

# 1-2. 리스트 슬라이싱
a = [1,2,3,4,5]
print(a[:2]) # [1,2]

# 1-3. 리스트 더하기
a = [1,2,3] 
b = [4,5,6]
print(a+b) # [1, 2, 3, 4, 5, 6]

c = [7,8,9,10]
print(a+c) # [1, 2, 3, 7, 8, 9, 10]
print(a*3) # [1, 2, 3, 1, 2, 3, 1, 2, 3]
# print(a[2] + 'hi') # TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(str(a[2]) + 'hi') # 3hi

f = '5'
# print(a[2] + f) # TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(a[2] + int(f)) # 8 

# 리스트 관련 함수
a = [1,2,3]
a.append(4)
print(a) # [1,2,3,4]

# a = a.append(5) #  print(a) -> None
# b = a.append(5) # AttributeError: 'NoneType' object has no attribute 'append'

a = [[1,2],[2,3],[3,4],[4,5]]
# print("a[:,0] : ", a[: ,0])
a.sort()
print(a) # [1, 2, 3, 4]
# print("a.shape", a.shape) # [1, 2, 3, 4] # shape는 numpy문법 

a.reverse()
print(a) # [4, 3, 2, 1]
print(a.index(2)) # 2이라는 값의 인덱스를 찾는다. -> 1
print(a.index(1)) # 

a.insert(0,7)
print(a) # [7, 4, 3, 2, 1]
a.insert(3,3)
print(a) # [7, 4, 3, 3, 2, 1]
a.remove(7)
print(a) # [7, 4, 3, 2, 1]
a.remove(3)
print(a) # [4, 3, 2, 1]
