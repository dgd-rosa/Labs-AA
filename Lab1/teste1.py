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

# %% 3
array_x = np.load('data1_x.npy')
array_y = np.load('data1_y.npy')
N = len(array_x)
P = 2
beta, X = LSestimation(N, P, array_x, array_y)

y_calc = np.matmul(X, beta)
plt.plot(array_x, y_calc,'r')
plt.plot(array_x, array_y, 'b*')
plt.show()

error = y_calc - array_y    
sse = np.matmul(np.transpose(error), error)

# %%4

array_x = np.load('data2_x.npy')
array_y = np.load('data2_y.npy')

N = len(array_x)
P = 2
beta, X = LSestimation(N, P, array_x, array_y)

y_calc = np.matmul(X, beta)
plt.plot(array_x, y_calc,'r^')
plt.plot(array_x, array_y, 'b*')
plt.show()

error = y_calc - array_y    
sse = np.matmul(np.transpose(error), error)
#SSE=1.342 -> GRANDE devido ao ruído?!

# %% 5
array_x = np.load('data2a_x.npy')
array_y = np.load('data2a_y.npy')

N = len(array_x)
P = 2
beta, X = LSestimation(N, P, array_x, array_y)

y_calc = np.matmul(X, beta)
plt.plot(array_x, y_calc,'r^')
plt.plot(array_x, array_y, 'b*')
plt.show()

error = y_calc - array_y    
print(np.matmul(np.transpose(error), error))
# dont know exactly what are outliers and if they need to be removed à la pata

# %% Regularization
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

data3_x = np.load('data3_x.npy')
data3_y = np.load('data3_y.npy')
alpha_arr = np.arange(0.001, 10, 0.01)
coefs = []

beta, X = LSestimation(len(data3_x), 2, data3_x, data3_y)
beta_aux = np.tile(np.transpose(beta),(len(alpha_arr),1))

for a in alpha_arr:
    ridge = linear_model.Ridge(alpha=a, max_iter= 10000)
    ridge.fit(data3_x, data3_y)
    if a == 0.001:
        coefs = ridge.coef_
    else:
        coefs = np.r_[coefs, ridge.coef_]

# DUVIDA: Usa-se B0, B1 e B2 ou B1,B2,B3
plt.figure()
plt.plot(alpha_arr, beta_aux, linestyle='--')
plt.plot(alpha_arr, coefs)
plt.legend(['B1 LS', 'B2 LS', 'B3 LS', 'B1 Ridge', 'B2 Ridge', 'B3 Ridge'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Regularization - Ridge Method')
plt.axis('tight')
plt.show()





