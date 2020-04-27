# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import GridSearchCV

## CSV読み込み
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
# ARR = []
# for i in range(1,13,2) :
#     for j in range(1,16,1) :
#         ARR.append([0,'Takakeisho','Kakuryu', 2020, i, j])

# test  = pd.DataFrame(ARR, columns=['w_judge','w_name','e_name','year','month','day'])

### 欠損値の削除
train = train.dropna()
test = test.dropna()
train = train.drop(columns=['e_judge'])
train = train.drop(columns=['ruler'])
train = train.drop(columns=['w_rank'])
train = train.drop(columns=['e_rank'])
test = test.drop(columns=['e_judge'])
test = test.drop(columns=['ruler'])
test = test.drop(columns=['w_rank'])
test = test.drop(columns=['e_rank'])

# データセットを結合
train = pd.concat([train,test], ignore_index=True)

### Category Encorder
for column in ['w_judge']:
    le = LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    le.fit(test[column])
    test[column] = le.transform(test[column])

### OneHot Encording
oh_w_class = pd.get_dummies(train.w_name)
oh_e_class = pd.get_dummies(train.e_name)
train.drop(['w_name','e_name'], axis=1, inplace=True)
train = pd.concat([train,oh_w_class,oh_w_class], axis=1)
_, i = np.unique(train.columns, return_index=True)
train = train.iloc[:, i]

### データセットの作成 (説明変数 -> X, 目的変数 -> Y)
X = train.drop('w_judge', axis=1)
y = train.w_judge
# print('X shape: {}, y shape: {}'.format(X.shape, y.shape))

### データセットの分割
# X_train = X[0:len(X)-len(ARR)]
# X_test = X.tail(len(ARR))
# y_train = y[0:len(y)-len(ARR)]
# y_test = y.tail(len(ARR))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# print("LinearRegression")
# model = LinearRegression()
# model.fit(X_train,y_train)
# print(model.score(X_train,y_train))
#
# print("LogisticRegression")
# model = LogisticRegression()
# model.fit(X_train,y_train)
# print(model.score(X_train,y_train))

# print("SVM")
# model = SVC()
# model.fit(X_train, y_train)
# predicted = model.predict(X_test)
# print(metrics.accuracy_score(Y_test, predicted))
# print("GridSearch")
# best_score = 0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         print(str(gamma) + "," + str(C))
#         svm = SVC(gamma=gamma, C=C)
#         svm.fit(X_train, y_train.values.ravel())
#         score = svm.score(X_test, y_test)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C':C, 'gamma':gamma}
# print("Best score: " + str(best_score))
# print("Best parameters: " + str(best_parameters))

# print("RandomForest")
# model = RandomForest(n_estimators=100).fit(X_train, y_train)
# print(model.score(X_test, y_test))

print("LightGBM")
train = lgb.Dataset(X_train, y_train)
test = lgb.Dataset(X_test, y_test, reference=train)
params = {
        'objective': 'binary',
        'metric': 'auc',
        }
model = lgb.train(params, train, valid_sets=test)

with open('model.pickle', mode='wb') as f:
    pickle.dump(model, f)

with open('model.pickle', mode='rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
print(len(y_pred))

acc = 0
for i in range(len(y_pred)) :
    acc = acc + y_pred[i]
print(acc / len(y_pred))


# model.save_model('model.txt')
#bst = lgb.Booster(model_file='model.txt')
#ypred = bst.predict(X_test, num_iteration=bst.best_iteration)
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
# auc = metrics.auc(fpr, tpr)
# print(auc)
