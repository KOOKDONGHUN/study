import numpy as np
import pandas as pd
from numpy import nan as NA

sample_data_frame = pd.DataFrame(np.random.rand(10,4))
print(f'{"-"*33}\n sample_data_frame : \n{sample_data_frame}\n')

# insert missing values
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5: , 3] = NA
print(f'{"-"*33}\n sample_data_frame : \n{sample_data_frame}\n')

print(f'{"-"*33}\n listwise deletion (dropna)!!\n')

sample_data_frame2 = sample_data_frame.dropna()
print(f'sample_data_frame2 : \n{sample_data_frame2}\n')
 
# 0, 1, 2 열 중에서 nan이 있는 행을 삭제하겠다
print(f'{"-"*33}\n pairwise deletion ([[0, 1, 2]]col dropna)!!\n')

sample_data_frame3 = sample_data_frame[[0, 1, 2]].dropna()
print(f'sample_data_frame3 : \n{sample_data_frame3}\n')

np.random.seed(0)
sample_data_frame4 = pd.DataFrame(np.random.rand(10,4))

sample_data_frame4.iloc[1, 0] = NA
sample_data_frame4.iloc[2, 2] = NA
sample_data_frame4.iloc[5: , 3] = NA

# 0열과 2열을 남기고 Nan을 포함하는 행을 삭제하고 출력하라
sample_data_frame4 = sample_data_frame4[[0,2]].dropna() 

print(f'{"-"*33}\n sample_data_frame4 : \n{sample_data_frame4}\n')

