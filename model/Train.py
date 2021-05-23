import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

def trainModel():
    # Upload Data 
    #xlsx = pd.ExcelFile('ITU-AI-Challenge-Turkcell-RLF-training-data-v1.3.xlsx')
    #sheets = pd.read_excel(xlsx, sheet_name=None, index_col=0, 
     #                      na_filter=True, convert_float=False)
    
    #rl_kpis = sheets['rl-kpis']
    #rl_kpis_work=rl_kpis[["severaly_error_second","error_second","unavail_second","avail_time","bbe","rlf"]]
   
    #x=rl_kpis_work.drop(columns=['rlf'])
    #y=rl_kpis_work['rlf']

    
    #rfc=RandomForestClassifier()
    #rfc.fit(x,y)
    #p=rfc.predict(x)
    
    #s=accuracy_score(y,p)
    #joblib.dump(rfc, 'iris-model.model')
    #print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s))

    iris_df = datasets.load_iris()

    x = iris_df.data
    y = iris_df.target

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dt = DecisionTreeClassifier().fit(X_train, y_train)
    preds = dt.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    joblib.dump(dt, 'iris-model.model')
    print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy))
    
        