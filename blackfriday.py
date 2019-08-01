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

target = 'Purchase'
features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
            'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1']

# missing_cols = [col for col in base.columns if base[col].isnull().any()]
# We will remove missing values columns

# Separate features and target
X = base[features].copy()
y = base.Purchase.copy()


# Finding categorical features
# s = (X.dtypes == 'object')
# object_cols = list(s[s].index)

# Treat categorical features, in order, using label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

X['Age'] = encoder.fit_transform(X['Age'])
X['Gender'] = encoder.fit_transform(X['Gender'])
X['Product_ID'] = encoder.fit_transform(X['Product_ID'])
X['City_Category'] = encoder.fit_transform(X['City_Category'])
X['Stay_In_Current_City_Years'] = encoder.fit_transform(X['Stay_In_Current_City_Years'])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print('MAE LinearRegression {}'.format(score_dataset(regressor, X_train, X_test, y_train, y_test)))

from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(random_state=1, n_estimators=30)
print('MAE RandomForestRegressor {}'.format(score_dataset(random_regressor, X_train, X_test, y_train, y_test)))
