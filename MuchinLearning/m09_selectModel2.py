import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import sklearn
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv',sep=',',header=1)

print(boston)

x = boston.iloc[:, :13].values
y = boston.iloc[:, 13].values

print(x)
print(y)

# y = y.reshape(-1)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=33)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

allAlgorithms = all_estimators(type_filter='regressor') # 

for (name, algorithm) in allAlgorithms:
    model = algorithm() 
    print(sklearn.__version__)
    try : 
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        # print(name,"의 정답률 = ", accuracy_score(y_test,y_pred))
        print(name,"의 정답률 = ", model.score(x_test,y_test))
    except:
        print("Error!!",name)
print(sklearn.__version__)