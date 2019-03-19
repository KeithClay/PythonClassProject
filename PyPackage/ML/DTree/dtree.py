# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sklearn
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r'C:\Users\keith\Documents\ML\HW\PyPackage\ML\data\wine.csv')
#print(data.head())
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

feature_list = data.columns

def dtree():
   
    print('Thank you!  You are using the Decision Tree model.\n')     
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    dt=DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    predictions=dt.predict(x_test)
    print("The Decision Tree score is ", round(dt.score(x_test, y_test)*100,2))
    errors=abs(predictions-y_test)
    print("Mean Abs error is",round(np.mean(errors),2))
    mape=np.mean(100*errors/y_test)    
    accuracy=100-mape
    print("Accuracy of the Decision Tree is",round(accuracy,2))
    
    importances=list(dt.feature_importances_)
    importances_list=[(data,round(imp,2)) for data,imp in zip(feature_list,importances)]
    feature_importances=sorted(importances_list,key=lambda x:x[1],reverse=True)
    print('The list of features in order of importance are ',feature_importances)
    
def dtree_CrossVal():
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    clf_scores=[]
    clf_s=[]
    for i in range(1,10):
        model_clf=DecisionTreeClassifier(max_depth=i)
        scores =model_selection.cross_val_score(model_clf, x_train, y_train, cv=10, scoring='accuracy')
        model_clf.fit(x_train,y_train)
        clf_scores.append((i,scores.mean()))
        clf_s.append(scores.mean())
        print("Depth range and accuracy score in Decision Tree is ",clf_scores)
        print('Length of list', len(clf_scores))
        print('Max of predicted scores ', max(clf_s))

        print("Thank you for using the Decision Tree model.")