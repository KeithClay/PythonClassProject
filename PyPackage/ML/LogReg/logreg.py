# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:56:20 2019

@author: keith
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sklearn
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data = pd.read_csv(r'C:\Users\keith\Documents\ML\HW\PyPackage\ML\data\wine.csv')
print(data.head())
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values


def logreg():
    
    test_s=int(input("How Much you want in testing data(5-100) ? "))
    x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=test_s/100)
    
    
    #x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
    lm = LogisticRegression()
    lm.fit(x_train,y_train)
    lm.predict(x_test)
    print('The predicted accuracy score is ',round(lm.score(x_test, y_test)*100,2))
      
    
  
    
    
    
    

   

    