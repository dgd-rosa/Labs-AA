# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:39:26 2020

@author: danie and gg
"""

import numpy as np
import matplotlib.pyplot as plt

#2
def preProcess(x):
    new_x = np.copy(x)
    if len(x.shape) == 1:
        new_x = np.copy(x).reshape(-1,1)
    print(new_x.shape)
    cols = np.size(new_x, 1)
    
    for col in range(cols):
        x_avg = np.average(new_x[:,col])
        new_x[:,col] = np.subtract(new_x[:,col], x_avg)
        
    return new_x
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
P = 1
#array_x = preProcess(array_x)
#array_y = preProcess(array_y)
beta, X = LSestimation(N, P, array_x, array_y)

y_calc = np.matmul(X, beta)
plt.plot(array_x, y_calc,'r')
plt.plot(array_x, array_y, 'b*')
plt.show()

error = y_calc - array_y    
sse = np.matmul(np.transpose(error), error)
print("Question 3: " , beta , "SSE: " , sse)
# %%4

array4_x = np.load('data2_x.npy')
array4_y = np.load('data2_y.npy')

N = len(array4_x)
P = 2
#array4_x = preProcess(array4_x)
#array4_y = preProcess(array4_y)
beta, X = LSestimation(N, P, array4_x, array4_y)

y_calc = np.matmul(X, beta)
plt.plot(array4_x, y_calc,'r^')
plt.plot(array4_x, array4_y, 'b*')
plt.show()

error = y_calc - array4_y    
sse = np.matmul(np.transpose(error), error)
#SSE=1.342 -> GRANDE devido ao ruído?!

print("Question 4: " , beta , "SSE: " , sse)

# %% 5
array5_x = np.load('data2a_x.npy')
array5_y = np.load('data2a_y.npy')

N = len(array5_x)
P = 2
array5_x = preProcess(array5_x)
array5_y = preProcess(array5_y)
beta, X = LSestimation(N, P, array5_x, array5_y)

y_calc = np.matmul(X, beta)
plt.plot(array5_x, y_calc,'r^')
plt.plot(array5_x, array5_y, 'b*')
plt.show()

error = y_calc - array5_y    
sse = np.matmul(np.transpose(error), error)
# dont know exactly what are outliers and if they need to be removed à la pata
print("Question 5: " , beta , "SSE: " , sse)


# %%5 without outliers
def retireOutliers(array_x, array_y):
    new_arrayx = []
    new_arrayy = []
    i = 0
    while i < len(array_x):
        if abs(array_y[i]) <= 1.5:
            new_arrayx.append(array_x[i])
            new_arrayy.append(array_y[i])
        i+=1
    return new_arrayx, new_arrayy
    
    
array5_x = np.load('data2a_x.npy')
array5_y = np.load('data2a_y.npy')


new_arrayx, new_arrayy = retireOutliers(array5_x, array5_y)
N = len(new_arrayx)
P = 2
beta, X = LSestimation(N, P, new_arrayx, new_arrayy)

y_calc = np.matmul(X, beta)
plt.plot(new_arrayx, y_calc,'r^')
plt.plot(new_arrayx, new_arrayy, 'b*')
plt.show()

error = y_calc - new_arrayy  
sse = np.matmul(np.transpose(error), error)
print("Question 5.2: " , beta , "SSE: " , sse)

# %% Regularization
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

data3_x1 = np.load('data3_x.npy')
data3_y1 = np.load('data3_y.npy')
alpha_arr = np.arange(0.001, 10, 0.01)
coefs = []

data3_x = preProcess(data3_x1)
data3_y = preProcess(data3_y1)

for a in alpha_arr:
    ridge = linear_model.Ridge(alpha=a, max_iter= 10000)
    ridge.fit(data3_x, data3_y)
    
    lasso = linear_model.Lasso(alpha=a, max_iter = 10000)
    lasso.fit(data3_x, data3_y)
    
    #Save the best lasso alfa
    #DUVIDA: Perguntar se o melhor alfa é o 1o para o qual 1 dos coefs do lasso é 0 ou aquele q tem mais coefs a 0?
    
    if a == 0.001:
        ridge_coefs = ridge.coef_
        lasso_coefs = lasso.coef_
    else:
        ridge_coefs = np.r_[ridge_coefs, ridge.coef_]
        lasso_coefs = np.c_[lasso_coefs, lasso.coef_]

lasso_coefs = lasso_coefs.transpose()
beta_lasso = lasso_coefs[0]
beta_ridge = ridge_coefs[0]
beta_lasso = np.tile(np.transpose(beta_lasso),(len(alpha_arr),1))
beta_ridge = np.tile(np.transpose(beta_ridge),(len(alpha_arr),1))
# DUVIDA: Usa-se B0, B1 e B2 ou B1,B2,B3
plt.figure()
plt.plot(alpha_arr, ridge_coefs)
plt.plot(alpha_arr, beta_ridge[:,0], linestyle='--', color='blue')
plt.plot(alpha_arr, beta_ridge[:,1], linestyle='--', color='orange')
plt.plot(alpha_arr, beta_ridge[:,2], linestyle='--', color='green')
plt.legend(['B1 Ridge', 'B2 Ridge', 'B3 Ridge', 'B1 LS', 'B2 LS', 'B3 LS'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Regularization - Ridge Method')
plt.axis('tight')
plt.show()

#Lasso Plot
plt.figure()
plt.plot(alpha_arr, beta_lasso[:,0], linestyle='--', color='blue')
plt.plot(alpha_arr, beta_lasso[:,1], linestyle='--', color='orange')
plt.plot(alpha_arr, beta_lasso[:,2], linestyle='--', color='green')
plt.plot(alpha_arr, lasso_coefs)
plt.legend(['B1 LS', 'B2 LS', 'B3 LS', 'B1 Lasso', 'B2 Lasso', 'B3 Lasso'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Regularization - Lasso Method')
plt.axis('tight')
plt.show()
#alpha=0.1 =>>> MELHOR ALPHA

# %% regularization 7
#choose alpha = 0.1
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
def LSestimation20(N, P, array_x, array_y, n_col):
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
            X[i,j] = array_x[i,n_col]**j
            j += 1
        i += 1
        
    
    aux1 = np.linalg.inv(np.matmul(np.transpose(X), X))
    
    beta = np.matmul(np.matmul(aux1, np.transpose(X)), array_y)
    
    return beta, X

data3_x1 = np.load('data3_x.npy')
data3_y1 = np.load('data3_y.npy')

data3_x = preProcess(data3_x1)
data3_y = preProcess(data3_y1)
N = len(data3_x)
P = 2

#LassoEstimation
best_alpha = 0.1
lasso = linear_model.Lasso(alpha=best_alpha, max_iter = 10000)
lasso.fit(data3_x, data3_y)
lasso_coefs = lasso.coef_

y_lasso_calc = np.matmul(lasso_coefs, np.transpose(data3_x))

scale = np.linspace(0, len(data3_x), num=50)
plt.figure()
plt.plot(scale, data3_y, color = 'blue')
plt.plot(scale, y_lasso_calc, color = 'red')

plt.title('Lasso Prediction VS Real Value')
plt.legend(['Data Y', 'Lasso Prediction'])
plt.show()

error = np.subtract(y_lasso_calc.reshape(-1,1), data3_y)
sse = np.matmul(np.transpose(error), error)
print("Question 7: SSE: " , sse)


