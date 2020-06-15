import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DataFrame의 널값 열별로 확인하기
def view_nan(data,print_view=True):
    for i in data.columns:
        # print(len(data[data[i].isnull()]))
        if print_view and len(data[data[i].isnull()]):
            print("missing values : ",i , "\t", len(data[data[i].isnull()]),"raw")



def split_x(seq, size=3):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)



def plot_feature_importances(model,data):
    n_features = len(data.columns)
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features),data.columns)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)

    plt.show()