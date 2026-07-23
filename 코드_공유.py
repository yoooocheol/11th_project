# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
wine_df = pd.DataFrame(wine.data, columns= wine.feature_names)

wine_df['target'] = wine.target

X = wine_df.drop(columns = 'target')
y = wine_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data_model = DecisionTreeClassifier(random_state=42)
data_parameters = {'criterion':['gini','entropy'], 'max_depth':[2,3,4,5], 'min_samples_split':[2,10],'min_samples_leaf':[1,2,4]}
data_grid = GridSearchCV(estimator= data_model, param_grid= data_parameters, scoring = 'accuracy', cv = 5)
data_grid.fit(X_train, y_train)
print('Best Hyper-parameter', data_grid.best_params_)
print('Best score', data_grid.best_score_)

data_best_model = data_grid.best_estimator_

data_importance = pd.DataFrame({'Feature':X.columns, 'importances':data_best_model.feature_importances_})
plt.figure(figsize=(12,5))
plt.bar(data_importance['Feature'], data_importance['importances'])
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('importances')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

