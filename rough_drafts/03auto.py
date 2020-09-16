'''

setting up autoencoder CLASS for kmeans on wild folder
code credit: https://github.com/alyserecord/wine/blob/master/src/cnn-autoencoder.py
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Sequential 

class Autoencoder():
    def __init__(self, model=None):
        ''' initializes the autoencoder model '''
        if model != None:
            self.model = model
    
    def build_autoencoder(self):
        #encoder layers
        autoencoder = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2,2), padding = 'same'),
            # layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            # layers.MaxPooling2D((2,2), padding = 'same'),
            # layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
            # layers.MaxPooling2D((2,2), padding = 'same'),
            # layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same'),
            # layers.MaxPooling2D((2,2), padding = 'same'),
            
            # layers.Flatten(),
            # layers.Reshape((4, 4, 8)), #reshape for next layer to deal with

            # layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same'),
            # layers.UpSampling2D((2,2)),
            # layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
            # layers.UpSampling2D((2,2)),
            # layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            # layers.UpSampling2D((2,2)),
            layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
            layers.UpSampling2D((2,2)),
            layers.Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')
            ])
        
        autoencoder.summary()

        autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

        self.model = autoencoder 

    def fit(self, train, test, batch_size, epochs):
        ''' this method will fit the model using train & test data '''
        self.model.fit(train, train, 
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data = (test, test))
        self.history = self.model.history.history

    def get_rmse(self, test):
        ''' calcuate the RMSE of the model after it is trained '''
        return self.model.evaluate(test, test)

    def predict(self, X):
        ''' 
        using the trained autoencoder 
        predicts on the provided image array and returns reconstructed images 
        ''' 
        return self.model.predict(X)

