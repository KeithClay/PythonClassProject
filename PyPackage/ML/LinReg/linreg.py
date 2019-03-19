# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:23:55 2019

@author: keith
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sklearn
from sklearn import linear_model
from regressors import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn import metrics


data = pd.read_csv(r'C:\Users\keith\Documents\ML\HW\PyPackage\ML\data\wine.csv')
#print(data.head())
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

def linreg():

    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    print("Estimated intercept coefficient: ",lr.intercept_)
    print("Number of coefficients: ", len(lr.coef_))
    print("Coefficients = ", lr.coef_)
        
    predicted = lr.predict(x_test)
    print("MSE is ", np.mean((y_test - predicted)**2))
    print ("MSE is ",sklearn.metrics.mean_squared_error(y_test, predicted))
    print ("Predicted array is",predicted)
    print("y_test is ", y_test)
    print("lrscore is ", lr.score(x_train, y_train))
    
    y_train_predict = lr.predict(x_train)
    r2 = r2_score(y_train, y_train_predict)
    print("r2 is ",r2)
    
    ols = linear_model.LinearRegression()
    z = ols.fit(x, y)
    
    print('coef_pval:\n', stats.coef_pval(ols, x, y))
    
    print(data.describe())
    
    sns.countplot(data['customer_segment'],label='Count')
    plt.show()
         
    corr = data.corr()
    plt.figure(figsize=(14,14))
    sns.heatmap(corr, cbar = True, square = True, cmap = 'coolwarm', annot=True)
    plt.show()
    
    plt.scatter(lr.predict(x_train), lr.predict(x_train)-y_train, c='g', s=30, alpha=0.5)
    plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test, c='b', s=30, alpha=0.5)
    plt.hlines(y=0, xmin = -5, xmax = 55)
    plt.title("Residuals")
    plt.ylabel("Residuals")