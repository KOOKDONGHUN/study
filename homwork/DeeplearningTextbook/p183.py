'''Numpy는 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 라이브러리 '''
import numpy as np
import time
from numpy.random import rand

# row size
n = 150

# reset metrics
matA = np.array(rand(n,n))
matB = np.array(rand(n,n))
matC = np.array([[0]*n for _ in range(n)])

#calc time
start = time.time()

# mul by for
for i in range(n):
    for j in range(n):
        for k in range(n):
            matC[i][j] = matA[i][k] * matB[k][j]

res1 = float(time.time())-start
# print(f"res : {float(time.time())-start}/s")
print(f"res1 : {res1}/s")

start = time.time()

matC = np.dot(matA,matB)
res2 = float(time.time())-start
print(f"res2 : {res2}/s")