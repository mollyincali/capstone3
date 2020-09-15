'''
ADJUSTED FROM 03CNN
-adding tensorboard callback
-changed from .image_dataset_from_dir to flow_from_dir
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
from keras.preprocessing.image import ImageDataGenerator

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
    img_size = 150

    train_datagen = ImageDataGenerator(
        train_data_path,
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    val_datagen = ImageDataGenerator(rescale = 1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=(img_size, img_size),
        batch_size = batch_size,
        class_mode = 'categorical')

    model = create_model((150, 150, 3), 3)

    model.compile(optimizer='adam', 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')

    es = EarlyStopping(monitor='val_loss', patience=2)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', 
                    histogram_freq=2, batch_size=batch_size)
    #command line: 
    #tensorboard --logdir=path_to_your_logs
    history = model.fit(train_generator,
        validation_data = val_generator,
        epochs = epoch,
        callbacks = [es, tensorboard])

    return history

if __name__ == "__main__":
    history = run_model(10)