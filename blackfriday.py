#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:45:23 2019

@author: elissandro
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.width', 100)
pd.set_option('precision', 3)

train = pd.read_csv("train.csv")

#y = train.iloc[:,11].values
#X = train.iloc[:,3].values
#correlacao = np.corrcoef(X, y)

train.hist()

plt.show()