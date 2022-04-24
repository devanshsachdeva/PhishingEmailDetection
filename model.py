# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:09:09 2019

@author: shubham
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data1 = pd.read_csv('Phishing_Dataset.txt',header = None)

x = data1.iloc[:,0:30].values
y = data1.iloc[:,30].values

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=0)

model = DecisionTreeClassifier(criterion='entropy',random_state=0)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
cm = confusion_matrix(ytest,y_pred)

model = RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0) 
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
cm = confusion_matrix(ytest,y_pred)

model = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
cm = confusion_matrix(ytest,y_pred)

model = SVC(kernel='rbf',random_state=0)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
cm = confusion_matrix(ytest,y_pred)

