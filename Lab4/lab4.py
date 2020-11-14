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
from scipy.stats import multivariate_normal

#For computing the mean and variance
def estimate_vals(x):
    return ( [ ( np.mean(column), np.var(column), len(column) ) for column in zip(*x) ] )


def estimate_covs(x):
    mean = []
    for column in zip(*x):
        mean.append(np.mean(column))
    return ( [  mean, np.cov(np.transpose(x)) ] )


#Calculate the mean and variance by class
#First we separe the input x according to y classes
def estimate_by_class(x, y, flag=0):
    class_division = dict()
    for i in range(len(x)):
        vector = x[i]
        if(y[i,0] not in class_division):
            class_division[y[i, 0]] = list()
        class_division[y[i, 0]].append(vector)
    
    #Now we compute the mean and variance for each class
    estimate = dict()
    if(flag == 1):
        for vals, rows in class_division.items():
            estimate[vals] = estimate_covs(rows)
    else:
        for vals, rows in class_division.items():
            estimate[vals] = estimate_vals(rows)
    return estimate


#Compute gaussian probabilities
def gaussian(x, mean, variance):
    exponent = exp(-( (x-mean)**2 / (2*variance) ))
    return ( 1 / sqrt(2*pi*variance)) * exponent
                        

def naive_bayes(x, estimates, total_rows):
    probs = dict()
    for val, params, in estimates.items():
        probs[val] = estimates[val][0][2] / float(total_rows)
        for i in range(len(params)):
            mean, var, _ = params[i]
            probs[val] *= gaussian(x[i], mean, var)
    return probs
            

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
#plt.show

#1.3
#First we compute the mean and variances
estimates = estimate_by_class(x_train, y_train)

max_prob = np.zeros((len(x_test), 1))
class_predict = np.zeros((len(x_test), 1))

for i in range(len(x_test)):
    naive = naive_bayes(x_test[i], estimates, len(x_train))
    #Check the max probability for knowing what class it is
    for prob in naive.items():
        if prob[1] > max_prob[i]:
            max_prob[i] = prob[1]
            class_predict[i] = prob[0]
            
#Predicted plot
for i in range(len(x_test)):
    if class_predict[i] == 1:
        plt.scatter(x_test[i, 0], x_test[i,1], color='red', label='1', marker='o')
    elif class_predict[i] == 2:
        plt.scatter(x_test[i, 0], x_test[i,1], color='green', label='1', marker='o')
    elif class_predict[i] == 3:
        plt.scatter(x_test[i, 0], x_test[i,1], color='blue', label='1', marker='o')

plt.show


acc_score = accuracy_score(y_test, class_predict)


#Now for the bayes method

estimate_bayes = estimate_by_class(x_train, y_train, 1)

max_prob = np.zeros((len(x_test), 1))
class_predict = np.zeros((len(x_test), 1))
bayes = np.zeros((len(x_test), 3))

for i in range(len(x_test)):
    for val, params in estimate_bayes.items():
        bayes[i, int(val)-1] = multivariate_normal.pdf(x_test[i], params[0], params[1])
    #Check the max probability for knowing what class it is
    for index in range(len(bayes[i])):
        if bayes[i, index] > max_prob[i]:
            max_prob[i] = bayes[i, index]
            class_predict[i] = index + 1
            

