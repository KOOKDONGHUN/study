import numpy as np

''' 2차원이상의 배열에? 2차원만 가능한 것인지는 모르겠음 궁금하긴 하나 3차원을 시도하는 것은 굉장히 귀찮다
    배열의 이름을 적고 행의 인덱스를 리스트 형태로 넣으면 그것이 fancy indexing 전혀 다른 새로운 ndarray를 생성한다.'''
arr = np.array([[1,2],[3,4],[5,6],[7,8]])
print(f"arr : \n{arr}\nnew_arr : \n{arr[[3,2,0]]}")

print("-"*33)

arr = np.arange(25).reshape(5,5)
print(f"arr : \n{arr}\nnew_arr : \n{arr[[1,3,0]]}")