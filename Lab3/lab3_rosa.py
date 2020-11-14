# -*- coding: utf-8 -*-
"""Lab3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Or2hbQzW-u_M0nEPYptEEtpjL70wTj8d

# Lab3 - Neural Network
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

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

# %%1.1
#dataset comes as (training_images, training_labels), (test_images, test_labels)

dataset = keras.datasets.fashion_mnist.load_data()

train_x = dataset[0][0]
train_y = dataset[0][1]

test_x = dataset[1][0]
test_y = dataset[1][1]

#1.1.2
#Mostra as primeiras 5 imagens de cada set
'''
for i in range(5):
    plt.figure()
    plt.imshow(train_x[i])
    plt.figure()
    plt.imshow(test_x[i])
'''
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
validation_x = np.expand_dims(validation_x, -1)

#print(train_x.shape) #(48000, 28, 28, 1)

#%%
"""**MLP**"""

#1.2.1
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
history = model.fit(x = train_x, y = train_y,batch_size=200, epochs=200,callbacks=callback, validation_data=(validation_x, validation_y))

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
predicted_y = model.predict(test_x)
predicted_y_val = model.predict(validation_x)

#Desired results vais ser igual, independentemente da NN utilizada
_, desired_results = np.where(test_y == test_y.max(axis=1)[:,None])
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

#%% *** WHITHOUT EARLY STOPPING ***


#1.2.1
model = keras.Sequential()
model.add(Flatten(input_shape = train_x[0].shape ))

#1.2.2 Add 2 hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))

#1.2.3 Add softmax layer
model.add(Dense(10, activation='softmax'))

#1.2.4
model.summary()

#1.2.6
#define compiler specifications
compiler_adam = Adam(learning_rate=0.001, clipnorm = 1)
#compile model
model.compile(optimizer=compiler_adam, loss='categorical_crossentropy')
#fit to model
history = model.fit(x = train_x, y = train_y,batch_size=200, epochs=200, validation_data=(validation_x, validation_y))

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
predicted_y = model.predict(test_x)
predicted_y_val = model.predict(validation_x)

#Desired results vais ser igual, independentemente da NN utilizada
_, desired_results = np.where(test_y == test_y.max(axis=1)[:,None])
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

#%% ******** CNN ********

#1.3.1
cnn = keras.Sequential()

#Add convolutional layer with 16 layers and another 16 layers
cnn.add(Conv2D( 16, (3,3), activation='relu', input_shape = train_x[0].shape ))

cnn.add( MaxPooling2D( pool_size=(2,2) ) )

cnn.add( Conv2D( 16, (3,3), activation='relu') )

cnn.add( MaxPooling2D( pool_size=(2,2) ) )

cnn.add( Flatten() )

cnn.add( Dense( 32, activation='relu' ) )

cnn.add( Dense( 10, activation='softmax' ) )

#1.3.2
cnn.summary()

callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


#1.3.3
#define compiler specifications
compiler_adam = Adam(learning_rate=0.001, clipnorm = 1)
#compile model
cnn.compile(optimizer=compiler_adam, loss='categorical_crossentropy')
#fit to model
history = cnn.fit(x = train_x, y = train_y,batch_size=200, epochs=200, callbacks=callback, validation_data=(validation_x, validation_y))

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
predicted_y = cnn.predict(test_x)
predicted_y_val = cnn.predict(validation_x)

#Desired results vais ser igual, independentemente da NN utilizada
_, desired_results = np.where(test_y == test_y.max(axis=1)[:,None])
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


#%% Printing
layers_idx = np.array([0,1,2,3])

plt.figure()
plt.imshow(test_x[4])
visualize_activations(cnn, layers_idx, test_x[4])
    