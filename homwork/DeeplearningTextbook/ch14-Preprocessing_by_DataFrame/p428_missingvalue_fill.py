import numpy as np
import pandas as pd
from numpy import nan as NA

np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10,4))
print(f'{"-"*33}\n sample_data_frame : \n{sample_data_frame}\n')

# insert missing values
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5: , 3] = NA
print(f'{"-"*33}\n sample_data_frame : \n{sample_data_frame}\n')

print(f'{"-"*33}\n fillna(0)!!\n')

sample_data_frame2 = sample_data_frame.fillna(0)
print(f'sample_data_frame2 : \n{sample_data_frame2}\n')

print(f'{"-"*33}\n fillna(method="ffill")!!\n')

sample_data_frame3 = sample_data_frame.fillna(method='ffill')
print(f'sample_data_frame3 : \n{sample_data_frame3}\n')

new_sample_data_frame = pd.DataFrame(np.random.rand(10,4))

new_sample_data_frame.iloc[1, 0] = NA
new_sample_data_frame.iloc[6: , 2] = NA

print(f'{"-"*33}\nnew_sample_data_frame : \n{new_sample_data_frame}\n')

print(f'fillna(method="ffill")!!\n')

# nan을 앞의 데이터로 채우시오
new_sample_data_frame = new_sample_data_frame.fillna(method='ffill')
print(f'new_sample_data_frame : \n{new_sample_data_frame}\n')


# nan을 열의 평균값으로 채우는 방법
sample_data_frame4 = sample_data_frame.fillna(sample_data_frame.mean())
