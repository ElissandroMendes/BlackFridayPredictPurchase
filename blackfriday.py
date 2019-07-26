#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:45:23 2019

@author: elissandro
"""

import pandas as pd
from matplotlib import pyplot as plt

base = pd.read_csv("train.csv")
data = base.drop(['User_ID', 'Product_ID', 'Product_Category_2', 'Product_Category_3'], axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, 7].values
y = y.reshape(-1, 1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 3] = encoder.fit_transform(X[:, 3])
X[:, 4] = encoder.fit_transform(X[:, 4])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
regressor.predict(X_test)