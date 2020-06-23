import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts,GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score,mean_absolute_error as mae

from xgboost import XGBRegressor,plot_importance


train = pd.read_csv("./data/dacon/comp1/train.csv",index_col=0,header=0,encoding="cp949")
test = pd.read_csv("./data/dacon/comp1/test.csv",index_col=0,header=0,encoding="cp949")
submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv",index_col=0,header=0,encoding="cp949")


train = train.loc[:, 'rho':'990_dst']
print(train)
test = test.loc[:, 'rho':'990_dst']

def outliners(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    upper_bound = q3+iqr*1.5
    lower_bound = q1-iqr*1.5
    return np.where((data>upper_bound) | (data<lower_bound))


cnt=0


# for i in range(len(train.columns)):

#     print("-idx-",i)

#     x=outliners(train.iloc[:,i])
#     print("-x-",x)
    
#     l = len(list(x[0]))
#     print("-len(x)-", l)
#     cnt+=l

# print(cnt)



# for i in range(len(train.columns)):
#     x=outliners(train.iloc[:,i])
#     l = len(list(x[0]))



train=train.transpose()
test=test.transpose()

train=train.interpolate()
test=test.interpolate() #보간법 // 선형보간

train=train.transpose()
test=test.transpose()

# print(train)

# print(outliners(train))

tmp_1ls = list()
for i in range(len(train.columns)):
    
    print("-idx-",i)

    x=outliners(train.iloc[:,i])
    print("-x-",x)
    
    l = len(list(x[0]))
    print("-len(x)-", l)
    # cnt+=l

    if l == 0 :
        tmp_1ls.append(i)

tmp_2ls = list()
for i in range(len(test.columns)):
    
    print("-idx-",i)

    x=outliners(test.iloc[:,i])
    print("-x-",x)
    
    l = len(list(x[0]))
    print("-len(x)-", l)
    # cnt+=l

    if l == 0 :
        tmp_2ls.append(i)

# print(cnt)

print(tmp_1ls)
print()
print(tmp_2ls)
print()
print(len(tmp_1ls))
print()
print(len(tmp_2ls))