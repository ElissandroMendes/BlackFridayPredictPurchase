#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:45:23 2019

@author: elissandro
"""

import pandas as pd
from matplotlib import pyplot as plt

base = pd.read_csv("train.csv")

base.drop(['Product_Category_2', 'Product_Category_3'], axis=1, inplace=True)
base.head()
base['Product_ID'].value_counts()
base.groupby('User_ID')['Product_ID'].count()



X = base.iloc[:, :-1].values

X['User_ID'].value_counts()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit()