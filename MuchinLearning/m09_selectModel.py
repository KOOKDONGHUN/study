import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.utils import all_estimators
from sklearn.utils.testing import all_estimators
import sklearn
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv',sep=',',header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=33)

allAlgorithms = all_estimators(type_filter='classifier') # iris에 대한 모든 모델링

for (name, algorithm) in allAlgorithms:
    model = algorithm() # -> 존나 희안한 문법인거 같은데 
    try :
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name,"의 정답률 = ", accuracy_score(y_test,y_pred))
    except : 
        print("Error!!",name)

print(sklearn.__version__)