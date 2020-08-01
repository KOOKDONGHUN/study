import torch

x = torch.ones(2,2, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*3
out = z.mean()
print(f'{z}\n{out}')

out.backward()
print(x.grad)

