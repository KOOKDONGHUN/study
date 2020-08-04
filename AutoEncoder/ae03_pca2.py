import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets['data']
y = datasets['target']

print(x.shape) # 442, 10
print(y.shape) # 442, 

# pca = PCA(n_components=5)
# x2 = pca.fit_transform(x)
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
# print(sum(pca_evr)) # 0.8340156689459766

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum) #  누적

n_components = np.argmax(cumsum >= 94) # 내가 지정한 숫자에 대한 인덱스 반환
n_components += 1