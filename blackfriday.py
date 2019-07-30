#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:45:23 2019

@author: elissandro
"""

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error

def score_dataset(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    predicts = model.predict(X_val)
    return mean_absolute_error(y_val, predicts)

# Load data
base = pd.read_csv("train.csv")
data = base.drop(['User_ID', 'Product_ID', 'Product_Category_2', 
                  'Product_Category_3', 'Gender', 'City_Category',
                  'Stay_In_Current_City_Years'], axis=1)

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values
# y = y.reshape(-1, 1)

# Treat categorical features, in order, using label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 1] = encoder.fit_transform(X[:, 1])
# X[:, 3] = encoder.fit_transform(X[:, 3])
# X[:, 4] = encoder.fit_transform(X[:, 4])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print('MAE LinearRegression {}'.format(score_dataset(regressor, X_train, X_test, y_train, y_test)))

from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(random_state=1, n_estimators=200)
print('MAE RandomForestRegressor {}'.format(score_dataset(random_regressor, X_train, X_test, y_train, y_test)))
