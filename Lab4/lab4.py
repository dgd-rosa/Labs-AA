# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:50:42 2020

@author: guilherme e daniel
"""

import numpy as np
from matplotlib import pyplot as plt
from math import exp
from math import pi
from math import sqrt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


#BICADO MEIO À TOA DA ISABEL

#For computing the mean and variance
def estimate_vals(x):
    return ( [ ( np.mean(column), np.var(column), len(column) ) for column in zip(*x) ] )

#Calculate the mean and variance by class
#First we separe the input x according to y classes
def estimate_by_class(x, y):
    class_division = dict()
    for i in range(len(x)):
        vector = x[i]
        if(y[i,0] not in class_division):
            class_division[y[i, 0]] = list()
        class_division[y[i, 0]].append(vector)
    
    #Now we compute the mean and variance for each class
    estimate = dict()
    for vals, rows in class_division.items():
        estimate[vals] = estimate_vals(rows)
    return estimate
            
            
            
#1.1 Load data
x_train = np.load('data1_xtrain.npy')
y_train = np.load('data1_ytrain.npy')
x_test = np.load('data1_xtest.npy')
y_test = np.load('data1_ytest.npy')

#1.2
#Train set plot
plt.figure()
plt.scatter(x_train[0:49, 0], x_train[0:49,1], color='red', label='1', marker='x')
plt.scatter(x_train[50:99, 0], x_train[50:99,1], color='green', label='2', marker='x')
plt.scatter(x_train[100:149, 0], x_train[100:149,1], color='blue', label='3', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-5, 8)
plt.ylim(-5, 8)
plt.legend()
plt.show

#Test set plot
plt.figure()
plt.scatter(x_test[0:49, 0], x_test[0:49,1], color='red', label='1', marker='x')
plt.scatter(x_test[50:99, 0], x_test[50:99,1], color='green', label='2', marker='x')
plt.scatter(x_test[100:149, 0], x_test[100:149,1], color='blue', label='3', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-5, 8)
plt.ylim(-5, 8)
plt.legend()
plt.show

#1.3
#First we compute the mean and variances
estimates = estimate_by_class(x_train, y_train)