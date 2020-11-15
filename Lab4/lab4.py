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
from sklearn.feature_extraction.text import CountVectorizer

#%%Starting
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


acc_score_nb = accuracy_score(y_test, class_predict)


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

plt.figure()
for i in range(len(x_test)):
    if class_predict[i] == 1:
        plt.scatter(x_test[i, 0], x_test[i,1], color='red', label='1', marker='o')
    elif class_predict[i] == 2:
        plt.scatter(x_test[i, 0], x_test[i,1], color='green', label='1', marker='o')
    elif class_predict[i] == 3:
        plt.scatter(x_test[i, 0], x_test[i,1], color='blue', label='1', marker='o')
     
        
acc_score_b = accuracy_score(y_test, class_predict)
#%% Part 3

x_en = pd.read_csv('en_trigram_count.tsv', sep='\t', header=None, index_col=0)
x_fr = pd.read_csv('fr_trigram_count.tsv', sep='\t', header=None, index_col=0)
x_es = pd.read_csv('es_trigram_count.tsv', sep='\t', header=None, index_col=0)
x_pt = pd.read_csv('pt_trigram_count.tsv', sep='\t', header=None, index_col=0)


X_train = np.zeros((4, len(x_en)))
X_train[0,:] = np.transpose(x_en[2])
X_train[1,:] = np.transpose(x_fr[2])
X_train[2,:] = np.transpose(x_es[2])
X_train[3,:] = np.transpose(x_pt[2])

Y_train = ['en', 'fr', 'es', 'pt']

#MultinomialNB does the Laplace Smoothing for default
naive_bayes = MultinomialNB(fit_prior=False, class_prior=[0.25, 0.25, 0.25, 0.25])

nb_fit = naive_bayes.fit(X_train, Y_train)

predictions = nb_fit.predict(X_train)

accuracy_scr = accuracy_score(Y_train, predictions)

cntVec = CountVectorizer(ngram_range=(3, 3), vocabulary=x_en[1], lowercase=True, analyzer='char')

phrases = ['Que fácil es comer peras.', 'Que fácil é comer pêras.', 'Today is a great day for sightseeing.', 'Je vais au cinéma demain soir.',
           'Ana es inteligente y simpática.', 'Tu vais à escola hoje.']


X_test = cntVec.fit_transform(phrases)

Y_test = ['es', 'pt', 'en', 'fr', 'es', 'pt']


predictions = nb_fit.predict(X_test)
accuracy_scr = accuracy_score(Y_test, predictions)


predict_prob = nb_fit.predict_proba(X_test)

sorted_predict = np.sort(predict_prob, axis=1)

margin = sorted_predict[:, -1] - sorted_predict[:, -2]
score = nb_fit.score(X_test, Y_test)

