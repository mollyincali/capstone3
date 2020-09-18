''' class for CNN model '''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import PIL 

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
        train_data_dir = '../animals/train'
        val_data_dir = '../animals/val'
        batch_size = batch_size

        train_datagen = ImageDataGenerator(
            # train_data_dir,
            rescale = 1. / 255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True)

        val_datagen = ImageDataGenerator(rescale = 1. / 255)
        
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(
            val_data_dir,
            target_size=img_size,
            batch_size = batch_size,
            class_mode = 'categorical')

        self.train_generator = train_generator
        self.val_generator = val_generator

    def fit_cnn(self, epoch):
        ''' fit the cnn model with modelcheckpoint as a callback '''

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = 'best_mod.hdf5',
            save_weights_only = True,
            monitor = 'val_acc',
            mode = 'max')

        self.model.fit(self.train_generator,
            validation_data = self.val_generator,
            epochs = epoch,
            callbacks = [checkpoint])

        self.history = self.model.history.history
    
    def predict(self, X):
        ''' predict on model '''
        return self.model.predict(X)

    def load_weights(self, weights_path):
        ''' load weights of previous model '''
        self.model.load_weights(weights_path)
        return self 

if __name__ == "__main__":
    cnn = CNN()
    cnn.build_cnn((150,150,3),3) 
    cnn.create_img_gen((150,150),32)    
    cnn.fit_cnn(1)     

    #get images the model guessed incorrectly
    x, y = next(cnn.val_generator)
    difference = np.argmax(cnn.predict(x), axis = 1) != np.argmax(y, axis = 1)
    diff = x[difference]
    np.sum(diff)
    for d in diff:
        img = (d * 255).astype(np.uint8)
        PIL.Image.fromarray(img).show();

'''
def graph_model(history, epochs):
    """ code to run accuracy on test and validation """
    pink = '#CC9B89'
    blue = '#23423F'
    gold = '#B68D34'
    epochs = 20

    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    loss=history.history['loss'] 
    val_loss=history.history['val_loss'] 
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy',
            linewidth = 2, color = blue)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy',
            linewidth = 2, color = pink)
    plt.legend(loc='lower right')
    plt.ylim((0.70,1))
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss',
            linewidth = 2, color = blue)
    plt.plot(epochs_range, val_loss, label='Validation Loss',
            linewidth = 2, color = pink)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show();
'''

