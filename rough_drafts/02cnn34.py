'''
final val_acc =  0.3462 acc = 0.3537
code credit 
https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D

import pathlib

def create_model(input_size, num_classes):
    """ cnn code from keras website """
    nb_filters = 32
    num_classes = 3
    pool_size = (2, 2)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(nb_filters, kernel_size = (3,3), padding='valid', 
                    input_shape=(150,150), activation='relu'),

        layers.Conv2D(nb_filters, 3, activation='relu'), #missing kernel

        layers.Conv2D(nb_filters, 3, activation='relu'), #missing kernel
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='softmax'), #higher dense
        layers.Dense(num_classes)
    ])
    return model 

def run_model():
    """    data generator & runs model returns history  """ 
    train_data_path = "../animals/train"

    batch_size = 32
    img_width = 150
    img_height = 150

    train_imgs = tf.keras.preprocessing.image_dataset_from_directory(train_data_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        validation_split=0.2,
        subset="training",
        batch_size=batch_size,
        image_size=(img_width, img_height),
        shuffle=True,
        seed=23)

    val_imgs = tf.keras.preprocessing.image_dataset_from_directory(train_data_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        validation_split=0.2,
        subset="validation",
        batch_size=batch_size,
        image_size=(img_width, img_height),
        shuffle=True,
        seed=23)

    model = create_model((150, 150, 3), 3)

    model.compile(optimizer='adam', 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')

    history = model.fit(train_imgs,
        validation_data=val_imgs,
        epochs=3)

    return history

if __name__ == "__main__":
    history = run_model()

