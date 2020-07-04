import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()

x = iris.data[:, (2,3)] # 꽃잎의 길이와 너비만 // x칼럼 2개만 로드
y = (iris.target == 0).astype(np.int) # Iris Setosa // 타겟값이 0인 친구들만 인트형식으로 로드

model = Perceptron()
model.fit(x,y)

y_pred = model.predict( [ [ 2, 0.5 ] ] )

print(y_pred)