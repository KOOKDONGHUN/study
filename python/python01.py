# 정수형
a = 1
b = 2
c = a + b
print(c) # 3
d = a * b
print(d) # 6
e = a / b
print(e) # 0.5 다른언어에서는 정수계산은 정수로만 나옴 

# 실수형
a = 1.1
b = 2.2
c = a + b
print(c) # 3.3000000000000003 ->  실수 연산시에는 컴퓨터 계산법에 부동소수점이라는 것 때문에 오차가 있음 
d = a * b
print(d) # 2.4200000000000004
e = a / b
print(e) # 0.5

# 문자형
a = 'hel'
b = 'lo'
c = a + b
print(c)

# 문자 + 숫자
a = 123
b = '45'
# c = a + b
# print(c)

# 숫자를 문자변환 + 변환
a = 123
a = str(a)
print(a)
c = a + b
print(c)

a = 123
b = '45'
b = int(b)
c = a + b
print(c)

#문자열 연산하기
a = 'abcdefgh'
print(a[0]) # a
print(a[3]) # d
print(a[5]) # f
print(a[-1]) # h
print(a[-2]) # g
print(type(a)) # <class 'str'>

b = 'xyz'
print(a+b) # abcdefghxyz

# 문자열 인덱싱 #뒤에서 시작할때는 -1하지 않음 
a = 'hello, Deep learning' 
print(a[7]) # D
print(a[-1]) # g
print(a[-2]) # n
print(a[3:9]) # lo, Dee 틀림 # lo, De 
print(a[3:-5]) # lo, Deep lea # 몰랐음 
print(a[:-1]) # hello, Deep learning 틀림 # hello, Deep learnin
print(a[1:]) # ello, Deep learning 
print(a[5:-4]) # , Deep learninggnin 틀림 # , Deep lear
