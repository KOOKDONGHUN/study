# 3. 딕셔너리 -> 중복 안됨
# {키 : 벨류}
# {key: value}

a = {1: 'hi', 2: 'hello'}
print(a) # {1: 'hi', 2: 'hello'}
print(a[1]) # hi

b = {'hi': 1, 'hello': 2}
print(b['hello']) # 2

# 딕셔너리 요소 삭제 
del a[1]
print(a) # {2: 'hello'}
del a[2]
print(a) # {}

a = { 1: 'a', 1: 'b', 1: 'c' } # 키 값은 중복된다면 가장 마지막에 써진 것만 인식 된다 
print(a) # {1: 'c'} 덮어써진다라는 느낌

b = {1: 'a', 2: 'a', 3: 'a'}  # 값은 중복이 되도 상관없다.
print(b) # {1: 'a', 2: 'a', 3: 'a'}

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys()) # dict_keys(['name', 'phone', 'birth'])
print(type(a.keys())) # <class 'dict_keys'>
print(a.values()) # dict_values(['yun', '010', '0511'])
print(type(a)) # <class 'dict'>
print(a.get('name')) # yun
print(a['name']) # yun
print(a.get('phone')) # 010
print(a['phone']) # 010
