# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics


data = pd.read_csv(r'C:\Users\keith\Documents\ML\HW\PyPackage\ML\data\wine.csv')
print(data.head())
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

print('The data columns by their order in the dataset are ',data.columns)
feature_list = data.columns

def ranfor():
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    rf=RandomForestRegressor(n_estimators=1000,random_state=40)
    rf.fit(x_train,y_train)
    predictions=rf.predict(x_test)
    print("The RF score is ", round(rf.score(x_test, y_test)*100,2))
    errors=abs(predictions-y_test)
    print("Mean Abs error is",round(np.mean(errors),2))
    mape=np.mean(100*errors/y_test)    
    accuracy=100-mape
    print("Accuracy of Random Forest is",round(accuracy,2))
    
    importances=list(rf.feature_importances_)
    importances_list=[(data,round(imp,2)) for data,imp in zip(feature_list,importances)]
    feature_importances=sorted(importances_list,key=lambda x:x[1],reverse=True)
    print('The list of features in order of importance are ',feature_importances)
    
    print('Thank you for using the Random Forest testing model!')