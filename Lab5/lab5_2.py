# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:05:04 2020

@author: ggmvi and rosa
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

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


warnings.filterwarnings('ignore', 'Solver terminated early.*')
#####################

def visualize_activations(conv_model,layer_idx,image):
    plt.figure(0)
    plt.imshow(image,cmap='gray')
    outputs = [conv_model.layers[i].output for i in layer_idx]
    
    visual = keras.Model(inputs = conv_model.inputs, outputs = outputs)
    
    features = visual.predict(np.expand_dims(np.expand_dims(image,0),3))  
        
    f = 1
    for fmap in features:
            square = int(np.round(np.sqrt(fmap.shape[3])))
            plt.figure(f)
            for ix in range(fmap.shape[3]):
                 plt.subplot(square, square, ix+1)
                 plt.imshow(fmap[0,:, :, ix], cmap='gray')
            plt.show()
            plt.pause(2)
            f +=1

    
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
    


# ####################### MLP ##########################
def mlp2(trainX, trainY, testX, testY, class_weight):
    #print("MLP")
    
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    #Dividir o training set em training set e validation set
    #stratify para que a percentagem de 0s e 1s no conjunto de validação seja igual à real
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.30, shuffle = True, stratify=trainY, random_state=0)
    
    mlp = Sequential()
    #Input layer
    inputShape = trainX[0].shape
    mlp.add(Dense(inputShape[0], activation='relu', input_shape=trainX[0].shape))
    
    #Hidden layers
    mlp.add(Dense(64, activation='relu'))
    mlp.add(Dense(128, activation='relu'))
    mlp.add(Dense(256, activation='relu')) 
    mlp.add(Dense(256, activation='relu'))
    mlp.add(Dense(128, activation='relu'))
    mlp.add(Dense(64, activation='relu'))
    
    #Output layer com apenas duas unidades: uma para a label 0 e outra para a label 1
    mlp.add(Dense(2, activation='softmax'))
    
    #Summary
    mlp.summary()
    
    #Compila o modelo
    adam_mlp = Adam(lr=0.00001, clipnorm=0.001)
    mlp.compile(optimizer=adam_mlp, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    #ES
    es_mlp = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights = True)
    
    #Faz fit do modelo aos dados de treino. Especifica o número máximo de iterações
    #em epochs (400), os dados da validação, e a função de callback es - Early Stopping
    mlp_history = mlp.fit(trainX, trainY, epochs=12800, validation_data=(valX, valY), callbacks=[es_mlp], batch_size=200,
                          class_weight=class_weight)
    
    #Faz plot da evolução da Loss no conjunto de treino e no conjunto de validação
    plt.figure()
    plt.plot(mlp_history.history['loss'], label='Training Set')
    plt.plot(mlp_history.history['val_loss'], label='Validation Set')
    plt.legend()
    plt.title('MLP with Early Stopping: Loss in Training Set VS Loss in Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    #Faz previsão para os dados do test set
    predicted_y_mlp = mlp.predict(testX)
        
    #Transforma as previsões num vetor coluna com o resultado da previsão
    _, predicted_results_mlp = np.where(predicted_y_mlp == predicted_y_mlp.max(axis=1)[:,None]) 
    
    return predicted_results_mlp, predicted_y_mlp



################################## MLP ########################################
        #   MinMax:
        #Accuracy Score: 0.8181818181818182
        #Balanced Accuracy Score: 0.7623339658444023
        #F Measure: 0.8732394366197184
        
# pred_y_mlp, pred_y_mlp_prob = mlp2(x2_train_mm, y2_train, x2_test_mm, y2_test, class_weight)
# metrics(y2_test, pred_y_mlp, pred_y_mlp_prob)

######################## MLP FUNCTION ########################


def mlp():
    print("MLP")


################################################
############# Dataset Real Estate ##############
x2_train = np.load("Real_Estate_Xtrain.npy", "r")
y2_train = np.load("Real_Estate_ytrain.npy", "r")
x2_test = np.load("Real_Estate_Xtest.npy", "r")
y2_test = np.load("Real_Estate_ytest.npy", "r")

count = np.zeros((2,1))
for i in range(np.size(y2_test)):
    if y2_test[i] == 1:
        count[0] += 1
    else:
        count[1] +=1
        
class_weight = dict()
class_weight[0] = 1.
class_weight[1] = 1.

#Normalizar features 
x2_train_norm = normalize(x2_train, axis=1)
x2_test_norm = normalize(x2_test, axis=1)
#Standardizar features
scaler = StandardScaler()
scaler.fit(x2_train, y2_train)
x2_train_std = scaler.transform(x2_train)
x2_test_std = scaler.transform(x2_test)
#Min max features
minmax = MinMaxScaler()
minmax.fit(x2_train, y2_train)
x2_train_mm = minmax.transform(x2_train)
x2_test_mm = minmax.transform(x2_test)

pred_y_mlp, pred_y_mlp_prob = mlp2(x2_train_mm, y2_train, x2_test_mm, y2_test, class_weight)

######################## MLP ########################
train_x, validation_x, train_y, validation_y = train_test_split(x2_train, y2_train, test_size=0.2, shuffle = False)

train_x = np.expand_dims(train_x, -1) #-1 to put the columns at the matrix end
x2_test = np.expand_dims(x2_test, -1)
validation_x = np.expand_dims(validation_x, -1)

model = keras.Sequential()
model.add(Flatten(input_shape = train_x[0].shape ))

#1.2.2 Add 2 hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))

#1.2.3 Add softmax layer
model.add(Dense(10, activation='softmax'))

#1.2.4
model.summary()

#1.2.5
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#1.2.6
#define compiler specifications
compiler_adam = Adam(learning_rate=0.001, clipnorm = 1)
#compile model
model.compile(optimizer=compiler_adam, loss='categorical_crossentropy')
#fit to model
history = model.fit(x = x2_train_mm, y = train_y,batch_size=200, epochs=200,callbacks=callback, validation_data=(validation_x, validation_y))

plt.figure()
plt.plot(history.history['loss'], label = 'Training Set')
plt.plot(history.history['val_loss'], label = 'Validation Set')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#1.2.8
#Testing set prediction
predicted_y = model.predict(x2_test)
predicted_y_val = model.predict(validation_x)

#Desired results vais ser igual, independentemente da NN utilizada
_, desired_results = np.where(y2_test == y2_test.max(axis=1)[:,None])
_, desired_results_val = np.where(validation_y == validation_y.max(axis=1)[:,None])

#Transform predictions shape
_, predicted_y = np.where(predicted_y == predicted_y.max(axis=1)[:,None])
_, predicted_y_val = np.where(predicted_y_val == predicted_y_val.max(axis=1)[:,None])

#Accuracy score
acc_score = accuracy_score(desired_results, predicted_y)
acc_score_val = accuracy_score(desired_results_val, predicted_y_val)

#Confusion matrix
conf_matrix = confusion_matrix(desired_results, predicted_y)

#Prints
print("\nMLP with Early Stopping")
print("Accuracy Score: {0}".format(acc_score))
print("Confusion Matrix:\n {0}".format(conf_matrix))

