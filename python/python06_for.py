a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys():
    print(i)

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:
    i = i*i
    print(i)

for i in a:
    print(i)

## while문
'''
while 조건문 :  #참인 동안 계속 반복한다.
    수행할 문장
'''
##### if문
if 1 :
    print("True")
else:
    print("False")
# "True"
if 3 :
    print("True")
else:
    print("False")
# "True"
if 0 :
    print("True")
else:
    print("False")
# "False"
if -1 :
    print("True")
else:
    print("False")
# "True"

'''
 비교 연산자
 <,>,==,!=,<=,>= 부등호
'''
a = 1
if a==1 :
    print("True")
else:
    print("False")
# "True"

money = 10000
if money >= 30000:
    print('한우를 먹는다.')
else :
    print('라면을 먹는다.') # 실행

### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print("한우먹자") # 실행
else :
    print('라면먹자') 

jumsu = [90,25,67,45,80]
number = 0
for i in jumsu:
    if i >= 60:
        print("경축 합격")
        number += 1

print("합격인원 ",number,"명") # 3명

##_____________________________##
# break, continue

jumsu = [90,25,67,45,80]
number=0

for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        print(" 경축 합격 ")
        number += 1
print("합격인원  %d명" %number) # 1명

jumsu = [90,25,67,45,80]
number=0
for i in jumsu:
    if i < 60:
        continue # 이후는 실행하지 않고 반복문을 처음 부터 다시 실행  -> 나는 그냥 무시하고 가는줄 pass
    if i >= 60:
        print(" 경축 합격 ")
        number += 1
print("합격인원  %d명" %number) # 3명