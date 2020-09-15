'''
cnn model for image classification
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

def create_model():
    """ final cnn model """

    return model

def data_gen():
    """ final data generator """
    mod = create_model()

    mod.compile(optimizer='adam', 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')
    
    history = mod.fit(train_generator, 
            validation_data=validation_generator,
            epochs=epochs)  

    return history

def graph_model(history, epochs):
    """ code to run accuracy on test and validation """
    epochs = 10
    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    loss=history.history['loss'] 
    val_loss=history.history['val_loss'] 
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show();

if __name__ == "__main__":
    pass

