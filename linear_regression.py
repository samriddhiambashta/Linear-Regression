# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:3].values
Y=dataset.iloc[:,-1]
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/10)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_predict=regressor.predict(X_test)
RD=float(input('Enter the R&D spending: '))
admin=float(input('Enter the administration cost: '))
market_cost=float(input('Enter the marketing cost: '))
user_data=np.array([RD,admin,market_cost])
profit=regressor.predict(user_data.reshape(1,-3))
print(profit)