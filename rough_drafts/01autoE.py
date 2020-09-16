''' 
ERROR with flow from dir
setting up autoencoder for kmeans on wild folder
trying with flow from dir
https://github.com/serengil/tensorflow-101/blob/master/python/ConvolutionalAutoencoder.ipynb
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
sns.set()

import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten

if __name__ == "__main__":
    img_width, img_height = 150, 150
    train_data_dir = '../animals/train/wild'
    val_data_dir = '../animals/val/wild'
    batch_size = 4

    train_datagen = ImageDataGenerator(
        train_data_dir,
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    val_datagen = ImageDataGenerator(rescale = 1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='input')

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        class_mode = 'input')

    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(16, (3, 3) 
        , padding='same', input_shape=(img_width,img_height,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    #2nd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #-------------------------

    #3rd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    
    #4rd convolution layer
    model.add(Conv2D(16,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #-------------------------

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation('sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    model.fit(train_generator, 
            epochs=3,
            validation_data=val_generator)
