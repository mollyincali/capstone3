'''
ADJUSTED FROM 02CNN
ERROR 
- using validation folder instead of subset of train
- some parameters on train_imgs & val_imgs commented out
- added kernel size
- increased epochs to 10
- early stopping
final val_acc =  0. acc = 0.
code copied from 
https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard

def create_model(input_size, num_classes):
    """ cnn code from keras website """
    nb_filters = 32
    num_classes = 3
    pool_size = (2, 2)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(nb_filters, kernel_size = (3,3), padding='valid', 
                    input_shape=input_size, activation='relu'),

        layers.Conv2D(nb_filters, kernel_size = (3,3), activation='relu'), #updated kernel

        layers.Conv2D(nb_filters, kernel_size = (3,3), activation='relu'), #updated kernel
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='softmax'), #higher dense
        layers.Dense(num_classes)
    ])
    return model 

def run_model(epoch):
    """    data generator & runs model returns history  """ 
    train_data_path = "../animals/train"
    val_data_path = "../animals/val"
    batch_size = 32

    train_imgs = tf.keras.preprocessing.image_dataset_from_directory(train_data_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        # validation_split=0.2,
        # subset="training",
        batch_size=batch_size,
        image_size=(150, 150),
        shuffle=True)
        # seed=23)

    val_imgs = tf.keras.preprocessing.image_dataset_from_directory(val_data_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        # validation_split=0.2,
        # subset="validation",
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=True)
        # seed=23)

    model = create_model((150, 150, 3), 3)

    model.compile(optimizer='adam', 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')

    es = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(train_imgs,
        validation_data = val_imgs,
        epochs = epoch,
        callbacks = [es])

    return history

if __name__ == "__main__":
    history = run_model(10)

