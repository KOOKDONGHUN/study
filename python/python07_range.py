# range 함수
a = range(10)
print(a) # range(0, 10)
b = range(1,11)
print(b) # range(1, 11)

for i in a:
    print(i)

for i in b:
    print(i)

print(type(a))

sum = 0
for i in range(1,11):
    sum += i
print(sum) # 55