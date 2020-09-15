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
    pass

def data_gen():
    """ final data generator """
    pass

def graph_model(history, epochs):
    """ code to run accuracy on test and validation """
    pink = '#CC9B89'
    blue = '#23423F'
    gold = '#B68D34'
    epochs = 10

    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    loss=history.history['loss'] 
    val_loss=history.history['val_loss'] 
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy',
            linewidth = 3, color = blue)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy',
            linewidth = 3, color = pink)
    plt.legend(loc='lower right')
    plt.ylim((0.70,1))
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss',
            linewidth = 3, color = blue)
    plt.plot(epochs_range, val_loss, label='Validation Loss',
            linewidth = 3, color = pink)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show();

if __name__ == "__main__":
    pass

