'''
ERROR: Unable to allocate 4.63 GiB for an array with shape (4739, 262144) and data type float32
setting up autoencoder for kmeans on wild folder
trying with numpy arrays
code credit: https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
sns.set()
import os
from skimage import color
from skimage import io

import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten

def get_val_array(rows):
    ''' turns validation images into numpy array for CNN '''
    array = np.zeros((rows, 262144))
    for idx, filename in enumerate(os.listdir('../animals/val/wild')):
        if filename != '.DS_Store':
            #turns into gray image
            image = color.rgb2gray(io.imread(f'../animals/val/wild/{filename}'))
            #flattens array so shape = 1, 262144
            image = image.reshape(1,512*512) 
            #add row to img_array
            array[idx,:] = image
    return array 

def get_train_array(rows):
    ''' turns training images into numpy array for CNN '''
    array = np.zeros((rows, 262144))
    for idx, filename in enumerate(os.listdir('../animals/train/wild')):
        if filename != '.DS_Store':
            #turns into gray image
            image = color.rgb2gray(io.imread(f'../animals/train/wild/{filename}'))
            #flattens array so shape = 1, 262144
            image = image.reshape(1, 512*512) 
            #add row to img_array
            array[idx,:] = image
    return array

def model_setup():
    ''' sets up autoencoder CNN '''
    pass

if __name__ == "__main__":
    val_array = get_val_array(500)
    train_array = get_train_array(4739)

    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(16, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
    , padding='same', input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    #2nd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    #here compressed version
    
    #3rd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    
    #4rd convolution layer
    model.add(Conv2D(16,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    model.fit(train_array, train_array, 
            epochs=3,
            validation_data=(val_array, val_array))