# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:39:07 2020

@author: Daniel ROsa e Guilherme Viegas
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

from sklearn.svm import SVC
from svm_plot import plot_contours
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

import warnings

from sklearn.metrics import plot_roc_curve

######
from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
#Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, roc_curve, auc
#Model selection
from sklearn.model_selection import GridSearchCV
#Dtree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
#import graphviz
#Feature Selection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
#import pydot
####

warnings.filterwarnings('ignore', 'Solver terminated early.*')
    
def metrics(trueY, predY, predYProb):
    print("Metrics in the test set")
    mlp_acc_score = accuracy_score(trueY, predY)
    print("Accuracy Score: {0}".format(mlp_acc_score))
    mlp_balanced_acc_score = balanced_accuracy_score(trueY, predY)
    print("Balanced Accuracy Score: {0}".format(mlp_balanced_acc_score))
    #F1 = 2 * (precision * recall) / (precision + recall)
    #precision = TP / (TP + FP)
    #recall = TP / (TP + FN)
    mlp_f_measure = f1_score(trueY, predY)
    print("F Measure: {0}".format(mlp_f_measure))
    #Confusion matrix
    mlp_confusion_matrix = confusion_matrix(trueY, predY)
    print("Confusion Matrix:")
    for i in range(2):
        for j in range(2):
            print(mlp_confusion_matrix[i][j], end=" ")
        print("\n", end="")
    return
    

# ############################## Naive Bayes ####################################
def naive_bayes(x1_train, y1_train, x1_test, y1_test, count):
    print("NAIVE BAYES\n")
    nb = GaussianNB()
    nb_model = nb.fit(x1_train, np.ravel(y1_train))
    predicted_nb_train = nb_model.predict(x1_train)
    nb_train_acc_score = accuracy_score(y1_train, predicted_nb_train)
    print("Accuracy Score in the training set: {0}".format(nb_train_acc_score))
    y_pred_nb = nb_model.predict(x1_test)
    y_pred_prob_nb = nb.predict_proba(x1_test)
    metrics(y1_test, y_pred_nb, y_pred_prob_nb)
    #plt.title('NB Dataset1')
    #plotting roc curve
    plot_roc_curve(nb_model,x1_test,y1_test)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve and AUC')
    plt.legend()

# ############################### SVM Poly ######################################
def svm_poly(x1_train, y1_train, x1_test, y1_test):   
    print("SVM with Polynomial kernel\n")
    parameters = {'kernel':['poly'], 'degree':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                  'max_iter':[100000], 'class_weight':['balanced','']}
    poly_svm = SVC()
    clf = GridSearchCV(poly_svm, parameters, scoring='accuracy')
    clf.fit(x1_train, np.ravel(y1_train))
    print("Best parameters: ", clf.best_params_)
    print("Best score: ", clf.best_score_)
    degree = clf.best_params_['degree']
    class_weight = clf.best_params_['class_weight']
    poly_svm = SVC(kernel='poly', degree=degree, class_weight=class_weight, max_iter=100000, probability=True)
    poly_svm.fit(x1_train, y1_train)
    print("Number of support vectors: ", np.sum(poly_svm.n_support_))
    poly_svm_train_pred_y = poly_svm.predict(x1_train)
    poly_train_acc = accuracy_score(y1_train, poly_svm_train_pred_y)
    print("Accuracy Score in the training set: ", poly_train_acc)
    poly_test_pred_y = poly_svm.predict(x1_test)
    poly_test_pred_prob_y = poly_svm.predict_proba(x1_test)
    metrics(y1_test, poly_test_pred_y, poly_test_pred_prob_y)
    #plotting roc curve
    plot_roc_curve(poly_svm,x1_test,y1_test)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve and AUC')
    plt.legend()

        
# ################################ SVM LINEAR ######################################
#Foram-se mudando os parâmetros e vendo os melhores e depois foi-se granularizando a procura
def svm_linear(x1_train, y1_train, x1_test, y1_test):    
    print("SVM with Linear kernel\n")
    start = -4
    end = 20
    num = 2*(end-start)
    cs = np.logspace(start=start, stop=end, num=num)
    cs = np.ndarray.tolist(cs)
    cs.append(float("inf"))
    parameters = {'kernel':['linear'], 'C': cs,'max_iter':[10000], 'class_weight':['balanced','']}
    lin_svm = SVC()
    clf = GridSearchCV(lin_svm, parameters, scoring='accuracy')
    clf.fit(x1_train, np.ravel(y1_train))
    print("Best parameters: ", clf.best_params_)
    print("Best score: ", clf.best_score_)
    lin_svm = SVC(kernel='linear', C = clf.best_params_['C'],
                  max_iter=100000, class_weight=clf.best_params_['class_weight'],
                  probability=True)
    lin_svm.fit(x1_train, np.ravel(y1_train))
    print("Number of support vectors: ", np.sum(lin_svm.n_support_))
    lin_train_pred_y = lin_svm.predict(x1_train) 
    lin_train_acc = accuracy_score(y1_train, lin_train_pred_y)
    print("Accuracy Score in the training set: ", lin_train_acc)
    lin_test_pred_y = lin_svm.predict(x1_test)
    lin_test_pred_prob_y = lin_svm.predict_proba(x1_test)
    metrics(y1_test, lin_test_pred_y, lin_test_pred_prob_y)
    #plotting roc curve
    plot_roc_curve(lin_svm,x1_test,y1_test)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve and AUC')
    plt.legend()

# ################################ SVM RBF ######################################
# #Foram-se mudando os parâmetros e vendo os melhores e depois foi-se granularizando a procura
def svm_rbf(x1_train, y1_train, x1_test, y1_test):   
    print("SVM with Gaussian RBF kernel\n")
    start = -5
    end = 5
    num = 2*(end-start)
    gammas = np.logspace(start=start, stop=end, num=num)
    gammas = np.ndarray.tolist(gammas)
    gammas.append('scale')
    start = -4
    end = 8
    num = 2*(end-start)
    cs = np.logspace(start=start, stop=end, num=num)
    cs = np.ndarray.tolist(cs)
    cs.append(float("inf"))
    parameters = {'kernel':['rbf'], 'C':cs, 'gamma':gammas, 'max_iter':[10000], 'class_weight':['balanced','']}
    rbf_svm = SVC()
    clf = GridSearchCV(rbf_svm, parameters, scoring='accuracy')
    clf.fit(x1_train, np.ravel(y1_train))
    print("Best parameters: ", clf.best_params_)
    print("Best score: ", clf.best_score_)
    rbf_svm = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], 
                  max_iter=10000, class_weight=clf.best_params_['class_weight'],
                  probability=True)
    rbf_svm.fit(x1_train, np.ravel(y1_train))
    print("Number of support vectors: ", np.sum(rbf_svm.n_support_))
    rbf_train_pred_y = rbf_svm.predict(x1_train) 
    rbf_train_acc = accuracy_score(y1_train, rbf_train_pred_y)
    print("Accuracy Score in the training set: ", rbf_train_acc)
    rbf_test_pred_y = rbf_svm.predict(x1_test)
    rbf_test_pred_prob_y = rbf_svm.predict_proba(x1_test)
    metrics(y1_test, rbf_test_pred_y, rbf_test_pred_prob_y)
    #plotting roc curve
    plot_roc_curve(rbf_svm,x1_test,y1_test)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve and AUC')
    plt.legend()





###########################################
############# Dataset Cancer ##############
x1_train = np.load("Cancer_Xtrain.npy", "r")
y1_train = np.load("Cancer_ytrain.npy", "r")
x1_test = np.load("Cancer_Xtest.npy", "r")
y1_test = np.load("Cancer_ytest.npy", "r")

count = np.zeros((2,1))
for i in range(np.size(y1_test)):
    if y1_test[i] == 1:
        count[0] += 1
    else:
        count[1] +=1


#svm_rbf(x1_train, y1_train, x1_test, y1_test)
#svm_linear(x1_train, y1_train, x1_test, y1_test)
#svm_poly(x1_train, y1_train, x1_test, y1_test)
#naive_bayes(x1_train, y1_train, x1_test, y1_test, count)



