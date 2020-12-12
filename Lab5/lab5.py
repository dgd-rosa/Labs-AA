# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:39:07 2020

@author: Daniel ROsa e Guilherme Viegas
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

############# Cancer Dataset ##############
x1_train = np.load("Cancer_Xtrain.npy", "r")
y1_train = np.load("Cancer_ytrain.npy", "r")
x1_test = np.load("Cancer_Xtest.npy", "r")
y1_test = np.load("Cancer_ytest.npy", "r")

count = np.zeros((2,1))
for i in range(np.size(y1_test)):
    if y1_test[i] == 0:
        count[0] += 1
    else:
        count[1] +=1

print("DATASET 1")