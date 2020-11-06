# -*- coding: utf-8 -*-

"""
Created on Fri Oct 30 18:15:04 2020

@author: ist

Usage: visualize_activations(CNN_model,[0,2],test_image)


"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split

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
            
            
# ********** OUR CODE **********

# %%1.1
#dataset comes as (training_images, training_labels), (test_images, test_labels)

dataset = keras.datasets.fashion_mnist.load_data()

train_x = dataset[0][0]
train_y = dataset[0][1]

test_x = dataset[1][0]
test_y = dataset[1][1]

#1.1.2
#Mostra as primeiras 5 imagens de cada set
for i in range(5):
    plt.figure()
    plt.imshow(train_x[i])
    plt.figure()
    plt.imshow(test_x[i])

#1.1.3
#Divide os elementos de todas as figuras oir 255 para ficarem com valores entre 0 e 1
#Usa-se true_divide pq o divide normal arrendonda para baixo, ou seja queremos por ex 4/5 = 0.75 em vez de 4/5=0
train_x = np.true_divide(train_x, 255)
test_x = np.true_divide(test_x, 255)

#1.1.4
train_y = keras.utils.to_categorical(train_y)
test_y = keras.utils.to_categorical(test_y)

#1.1.5
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.2, shuffle = False)

#1.1.6
#print(train_x.shape) #(48000, 28, 28)

train_x = np.expand_dims(train_x, -1) #-1 to put the columns at the matrix end
test_x = np.expand_dims(test_x, -1)

#print(train_x.shape) #(48000, 28, 28, 1)


# %% 1.2 MLP

