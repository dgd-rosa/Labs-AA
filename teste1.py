# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:39:26 2020

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt

#2
def LSestimation(N, P, array_x, array_y):
    X = [[0] * (P+1)] * N   
    X = np.array(X)
    X = X.astype(float) 
    array_x = np.array(array_x)
    i = 0
    j = 0
    #creating matrix X
    while i < N:
        j = 0
        while j <= P:
            X[i,j] = array_x[i,0]**j
            j += 1
        i += 1
        
    
    aux1 = np.linalg.inv(np.matmul(np.transpose(X), X))
    
    beta = np.matmul(np.matmul(aux1, np.transpose(X)), array_y)
    
    return beta, X

#3
array_x = np.load('data1_x.npy')
array_y = np.load('data1_y.npy')
N = 20
P = 1
beta, X = LSestimation(N, P, array_x, array_y)

y_calc = np.matmul(X, beta)
plt.plot(array_x, y_calc,'r')
plt.plot(array_x, array_y, 'b*')
plt.show()

error = y_calc - array_y
    
quadratic_error = np.matmul(np.transpose(error), error)