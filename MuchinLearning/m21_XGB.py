from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=3
)

# model = DecisionTreeClassifier(max_depth=4) # default? 몇이 좋냐고?
# model = RandomForestClassifier() # default? 몇이 좋냐고?
# model = GradientBoostingClassifier() # default? 몇이 좋냐고?
model = XGBClassifier() # default? 몇이 좋냐고?


'''
max_features : 기본값 써라!
n_estimators : 클수록 좋다! , 단점은 메모리를 많이 차지한다, 기본값 100
n_jobs = -1 : CPU 병렬 처리
'''

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)
print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)

    plt.show()

plot_feature_importances_cancer(model)