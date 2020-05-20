def sum1(c,d):
    return c+d

a = 1
b = 2
c = sum1(a,b)
print(c) # 3

###### 곱셈, 나눗셈, 뺄셈

def min(c,d):
    return c-d

def nanugi(c,d):
    return c/d

def gobhagi(c,d):
    return c*d

m = min(a,b)
n = nanugi(a,b)
g = gobhagi(a,b)

print(m)
print(n)
print(g)

def sayYeh():
    return 'Hi'
say = sayYeh()
print(say)


def sum2(c,d,e):
    return c+d+e

a = 1
b = 2
e = 3
c = sum2(a,b,e)
print(c) # 6