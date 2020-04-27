# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import numpy  as np
import lightgbm as lgb
import os
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

app = Flask(__name__)

def pred(n) :
    name = n.split(',')

    ## CSV読み込み
    train = pd.read_csv("data/train.csv")

    ARR1 = []
    for i in range(1,13,2) :
        for j in range(1,16,1) :
            ARR1.append([1,name[0],name[2], 2019, i, j])
    test1  = pd.DataFrame(ARR1, columns=['w_judge','w_name','e_name','year','month','day'])


    ### 欠損値の削除
    train = train.dropna()
    test1 = test1.dropna()
    train = train.drop(columns=['e_judge'])
    train = train.drop(columns=['ruler'])
    train = train.drop(columns=['w_rank'])
    train = train.drop(columns=['e_rank'])

    # データセットを結合
    train1 = pd.concat([train,test1], ignore_index=True)
    
    ### Category Encorder
    for column in ['w_judge']:
        le = LabelEncoder()
        le.fit(train1[column])
        train1[column] = le.transform(train1[column])
        le.fit(test1[column])
        test1[column] = le.transform(test1[column])

    ### OneHot Encording
    oh_w_class = pd.get_dummies(train1.w_name)
    oh_e_class = pd.get_dummies(train1.e_name)
    train1.drop(['w_name','e_name'], axis=1, inplace=True)
    train1 = pd.concat([train1,oh_w_class,oh_w_class], axis=1)
    _, i = np.unique(train1.columns, return_index=True)
    train1 = train1.iloc[:, i]

    ### データセットの作成 (説明変数 -> X, 目的変数 -> Y)
    X1 = train1.drop('w_judge', axis=1)
    y1 = train1.w_judge
    # print('X shape: {}, y shape: {}'.format(X.shape, y.shape))

    ### データセットの分割
    X1_train = X1[0:len(X1)-len(ARR1)]
    X1_test = X1.tail(len(ARR1))
    y1_train = y1[0:len(y1)-len(ARR1)]
    y1_test = y1.tail(len(ARR1))

    with open('model.pickle', mode='rb') as f:
        model = pickle.load(f)

    y_pred1 = model.predict(X1_test, num_iteration=model.best_iteration)

    acc1  = 0
    for i in range(len(y_pred1)) :
        acc1 = acc1 + y_pred1[i]
    acc1 = acc1 / len(y_pred1)

    if acc1 > 0.5:
        return acc1, name[0], name[1]
    else :
        return 1 - acc1, name[2], name[3]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        accuracy, name, jname = pred(request.form["vs"])
        print(accuracy,name,jname)
        return render_template('./result.html', result = round(accuracy*100) , winname = name, japanese = jname)

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
