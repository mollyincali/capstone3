'''
attempt to create CNN class
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard

class CNN():
    def __init__(self, model=None):
        ''' initializes the CNN model '''
        if model != None:
            self.model = model

    def build_cnn(self, input_size, n_categories):
        ''' build cnn architecture 
        
        input_size: tuple as (img_width, img_heigh, color_layer)
        n_categories: float how many classes predicting 
        ''' 
        nb_filters = 32
        kernel_size = (3, 3)
        pool_size = (2, 2)
        cnn = tf.keras.Sequential([
            layers.Conv2D(nb_filters, kernel_size = kernel_size,
                            padding='valid',
                            input_shape=input_size, 
                            activation = 'relu'),
            layers.Conv2D(nb_filters, kernel_size = kernel_size, 
                            activation = 'relu'),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation = 'relu'),
            layers.Dropout(0.5), 
            layers.Dense(n_categories, activation = 'softmax')
            ])

        cnn.summary()

        cnn.compile(optimizer='adam', 
                    loss ='categorical_crossentropy', 
                    metrics ='accuracy')
        
        self.model = cnn
    
    def create_img_gen(self, img_size, batch_size):
        ''' create image generators

        img_size: tuple
        batch_size: float 
        
        output: train_generator, val_generator
        '''
        img_width, img_height = img_size, img_size
        train_data_dir = '../animals/train'
        val_data_dir = '../animals/val'
        batch_size = batch_size

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
            class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(
            val_data_dir,
            target_size=(img_width, img_height),
            batch_size = batch_size,
            class_mode = 'categorical')

        self.train_generator = train_generator
        self.val_generator = val_generator

    def fit_cnn(self, epoch):
        ''' fit the cnn model with modelcheckpoint as a callback '''
#ISSUE HERE! 
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = 'best_mod.hdf5',
            save_weights_only = True,
            monitor = 'val_acc',
            mode = 'max')
        
        self.model.fit(self.train_generator,
            validation_data = self.val_generator,
            epochs = epoch,
            callbacks = [self.checkpoint])
            
        self.history = self.model.history.history

if __name__ == "__main__":
    pass