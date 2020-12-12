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
#NB
from sklearn.naive_bayes import GaussianNB
#SVM
from sklearn.svm import SVC
#Model selection
from sklearn.model_selection import GridSearchCV
#Dtree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
#import graphviz
#Feature Selection
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#import pydot
####

warnings.filterwarnings('ignore', 'Solver terminated early.*')

#Computing the best gamma for the svm
def searchGammas(x1_train, y1_train, x1_test, y1_test):
    #For utilizado para a ajudar pesquisa de gammas
    gammas = np.linspace(0.001, 1, 1000)
    sv_min = len(x1_test)
    min_gamma = 0.001
    
    for gamma in gammas:
        #max_iter = 100000 porque havia alguns gammas para os quais não convergia ou demorava muito
        chess_rbf_svm = SVC(kernel='rbf', C=float("inf"), gamma=gamma, max_iter=100000)
        chess_rbf_svm.fit(x1_train, y1_train.ravel())
        chess_pred_y = chess_rbf_svm.predict(x1_test)
        chess_error_percent = 1 - accuracy_score(y1_test, chess_pred_y.ravel())
        chess_sv_num = np.sum(chess_rbf_svm.n_support_)
        
        if (chess_error_percent == 0 and chess_sv_num < sv_min):
            sv_min = chess_sv_num 
            min_gamma = gamma
    print(min_gamma)
    
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
    #ROC curve e ROC area
    categoricalTrueY = to_categorical(trueY)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(categoricalTrueY[:,i], predYProb[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    colors = cycle(['aqua', 'darkorange'])
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    #Confusion matrix
    mlp_confusion_matrix = confusion_matrix(trueY, predY)
    print("Confusion Matrix:")
    for i in range(2):
        for j in range(2):
            print(mlp_confusion_matrix[i][j], end=" ")
        print("\n", end="")
    return
    

############# Dataset Cancer ##############
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

####### Instanciar Gaussian RBF SVM ########
# searchGammas(x1_train, y1_train, x1_test, y1_test)

# rbf_svm = SVC(kernel='rbf', gamma=3.414548873833601e-05, probability=True)

# #Treina o SVM
# rbf_svm.fit(x1_train, y1_train.ravel())

# #Obtém o predicted outcome para os dados de treino
# rbf_pred_y = rbf_svm.predict(x1_test)

# #Contornos das regiões de decisão 
# #plot_contours(rbf_svm, x1_test, y1_test.ravel())

# rbf_test_pred_prob_y = rbf_svm.predict_proba(x1_test)

# #Percentagem de erro
# rbf_error_percent = 1 - accuracy_score(y1_test.ravel(), rbf_pred_y)
# print(rbf_error_percent)
# #Número de support vectors
# rbf_sv_num = np.sum(rbf_svm.n_support_)

# # #TODO: Fazer métricas de accuracy
# metrics(y1_test, rbf_pred_y, rbf_test_pred_prob_y)

# ############################## Naive Bayes ####################################
#Acc = 58% no training set e 51% no test set. Not very good
##Standardizar features - acc entre 0.65 e 0.7
#scaler = StandardScaler()
#scaler.fit(x1_train, y1_train)
#x1_train = scaler.transform(x1_train)
#x1_test = scaler.transform(x1_test)
print("Naive Bayes")
nb = GaussianNB(count/np.sum(count))
nb_model = nb.fit(x1_train, y1_train.ravel())
predicted_nb_train = nb_model.predict(x1_train)
nb_train_acc_score = accuracy_score(y1_train, predicted_nb_train)
print("Accuracy Score in the training set: {0}".format(nb_train_acc_score))
y_pred_nb = nb_model.predict(x1_test)
y_pred_prob_nb = nb.predict_proba(x1_test)
metrics(y1_test, y_pred_nb, y_pred_prob_nb)
plt.title('NB Dataset1')

# ############################### SVM Poly ######################################
# parameters = {'kernel':['poly'], 'degree':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
#               'max_iter':[100000], 'class_weight':['balanced','']}
# poly_svm = SVC()
# clf = GridSearchCV(poly_svm, parameters, scoring='accuracy')
# clf.fit(x1_train, np.ravel(y1_train))
# print("Best parameters: ", clf.best_params_)
# print("Best score: ", clf.best_score_)
# degree = clf.best_params_['degree']
# class_weight = clf.best_params_['class_weight']
# poly_svm = SVC(kernel='poly', degree=degree, class_weight=class_weight, max_iter=100000, probability=True)
# poly_svm.fit(x1_train, y1_train)
# print("Number of support vectors: ", np.sum(poly_svm.n_support_))
# poly_svm_train_pred_y = poly_svm.predict(x1_train)
# poly_train_acc = accuracy_score(y1_train, poly_svm_train_pred_y)
# print("Accuracy Score in the training set: ", poly_train_acc)
# poly_test_pred_y = poly_svm.predict(x1_test)
# poly_test_pred_prob_y = poly_svm.predict_proba(x1_test)
# metrics(y1_test, poly_test_pred_y, poly_test_pred_prob_y)
# plt.title('Polynomial SVM Dataset1')

        
# ################################ SVM LINEAR ######################################
# #Foram-se mudando os parâmetros e vendo os melhores e depois foi-se granularizando a procura
# start = -5
# end = 3
# num = 2*(end-start)
# gammas = np.logspace(start=start, stop=end, num=num)
# gammas = np.ndarray.tolist(gammas)
# gammas.append('scale')
# start = -4
# end = 8
# num = 2*(end-start)
# cs = np.logspace(start=start, stop=end, num=num)
# cs = np.ndarray.tolist(cs)
# cs.append(float("inf"))
# parameters = {'kernel':['linear'], 'C':cs, 'gamma':gammas, 'max_iter':[10000], 'class_weight':['balanced','']}
# rbf_svm = SVC()
# clf = GridSearchCV(rbf_svm, parameters, scoring='accuracy')
# clf.fit(x1_train, np.ravel(y1_train))
# print("Best parameters: ", clf.best_params_)
# print("Best score: ", clf.best_score_)
# rbf_svm = SVC(kernel='linear', 
#               max_iter=10000, class_weight=clf.best_params_['class_weight'],
#               probability=True)
# rbf_svm.fit(x1_train, np.ravel(y1_train))
# print("Number of support vectors: ", np.sum(rbf_svm.n_support_))
# rbf_train_pred_y = rbf_svm.predict(x1_train) 
# rbf_train_acc = accuracy_score(y1_train, rbf_train_pred_y)
# print("Accuracy Score in the training set: ", rbf_train_acc)
# rbf_test_pred_y = rbf_svm.predict(x1_test)
# rbf_test_pred_prob_y = rbf_svm.predict_proba(x1_test)
# metrics(y1_test, rbf_test_pred_y, rbf_test_pred_prob_y)
# plt.title('LINEAR SVM Dataset1')

# ################################ SVM RBF ######################################
# #Foram-se mudando os parâmetros e vendo os melhores e depois foi-se granularizando a procura
# start = -5
# end = 3
# num = 2*(end-start)
# gammas = np.logspace(start=start, stop=end, num=num)
# gammas = np.ndarray.tolist(gammas)
# gammas.append('scale')
# start = -4
# end = 8
# num = 2*(end-start)
# cs = np.logspace(start=start, stop=end, num=num)
# cs = np.ndarray.tolist(cs)
# cs.append(float("inf"))
# parameters = {'kernel':['rbf'], 'C':cs, 'gamma':gammas, 'max_iter':[10000], 'class_weight':['balanced','']}
# rbf_svm = SVC()
# clf = GridSearchCV(rbf_svm, parameters, scoring='accuracy')
# clf.fit(x1_train, np.ravel(y1_train))
# print("Best parameters: ", clf.best_params_)
# print("Best score: ", clf.best_score_)
# rbf_svm = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], 
#               max_iter=10000, class_weight=clf.best_params_['class_weight'],
#               probability=True)
# rbf_svm.fit(x1_train, np.ravel(y1_train))
# print("Number of support vectors: ", np.sum(rbf_svm.n_support_))
# rbf_train_pred_y = rbf_svm.predict(x1_train) 
# rbf_train_acc = accuracy_score(y1_train, rbf_train_pred_y)
# print("Accuracy Score in the training set: ", rbf_train_acc)
# rbf_test_pred_y = rbf_svm.predict(x1_test)
# rbf_test_pred_prob_y = rbf_svm.predict_proba(x1_test)
# metrics(y1_test, rbf_test_pred_y, rbf_test_pred_prob_y)
# plt.title('RBF SVM Dataset1')