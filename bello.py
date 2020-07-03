d = 10000

bello = 5.55556
male = 1.38889
female = 4.16667

temp = 0

for i in range(10000):
    for j in range(10000):
        temp += bello*i + female*j

        if temp == d:
            print(i,j)