import torch
import numpy as np

x = torch.ones(2,2, requires_grad=True)
print(x)

y = x + 2
print(y)

