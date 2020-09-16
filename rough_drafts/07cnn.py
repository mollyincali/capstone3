''' 
final val_acc = 
UPDATING 06CNN
code credit: lecture + solutions
- adding model checkpoint
- cranking up epochs
- early stop changed to patience 2
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

def create_model(input_size, n_categories):
    """ baseline cnn code from solutions """
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    model = tf.keras.Sequential([
        layers.Conv2D(nb_filters, kernel_size = kernel_size,
                        padding='valid',
                        input_shape=input_size, 
                        activation = 'relu'),
        layers.Conv2D(nb_filters, kernel_size = kernel_size, 
                        activation = 'relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Dropout(0.25),
    # transition to an multi-layer perceptron
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.5), #more regularized 
        layers.Dense(n_categories, activation = 'softmax')
    ])
    return model

def run_model(epoch):
    """    data generator & runs model returns history  """ 
    img_width, img_height = 150, 150
    train_data_dir = '../animals/train'
    val_data_dir = '../animals/val'
    batch_size = 32

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

    mod = create_model((150, 150, 3), 3)

    low_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    mod.compile(optimizer=low_adam, 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')

    es = EarlyStopping(monitor='val_loss', 
                    patience=2)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = '../best_mod',
        save_weights_only = True,
        monitor = 'val_acc',
        mode = 'max',
        save_best_only = True)

    history = mod.fit(train_generator,
        validation_data = val_generator,
        epochs = epoch,
        callbacks = [es, model_checkpoint])

    return history

if __name__ == "__main__":
    history = run_model(100)
