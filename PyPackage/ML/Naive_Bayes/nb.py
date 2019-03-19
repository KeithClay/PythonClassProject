# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv(r'C:\Users\keith\Documents\ML\HW\PyPackage\ML\data\wine.csv')
#print(data.head())
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

def nb():
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    target_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, target_pred, normalize=True)
    accuracy=accuracy*100
    print('Accuracy of the Gaussian model is ', accuracy, "%")
    
    print('Thank you for using the Naive-Bayes model!')