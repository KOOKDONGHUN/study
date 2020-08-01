import torch

x = torch.ones(1, requires_grad=True)
print(x)

y = x+2
print(y)

print('====Run backward====')
y.backward()
print(y)

'''뭘 말하고자 하는 건지 잘 모르겠음'''

