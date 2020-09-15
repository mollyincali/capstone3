''' 
final val_acc = 0.9666 acc = 0.9555
code credit - lecture + solutions
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def create_model(input_size, n_categories):
    """ baseline cnn code from solutions """
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an multi-layer perceptron
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) #more regularized 
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
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

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical')

    mod = create_model((150, 150, 3), 3)

    mod.compile(optimizer='adam', 
                loss ='categorical_crossentropy', 
                metrics ='accuracy')

    es = EarlyStopping(monitor='val_loss', 
                    patience=5, 
                    restore_best_weights=True)

    history = mod.fit(train_generator, 
            validation_data = validation_generator,
            epochs = epoch,
            callbacks = [es])  

    return history

if __name__ == "__main__":
    history = run_model(10)

'''
results if want to update graph
val_loss = [0.23665618896484375,
 0.2028890997171402,
 0.18763041496276855,
 0.15617439150810242,
 0.15107859671115875,
 0.17908154428005219,
 0.13528262078762054,
 0.3348764479160309,
 0.12064114212989807,
 0.11799189448356628]

loss = [0.5892452001571655,
 0.26536187529563904,
 0.22190652787685394,
 0.1798446625471115,
 0.16781146824359894,
 0.15261150896549225,
 0.14150552451610565,
 0.13836263120174408,
 0.12002164870500565,
 0.12109742313623428]

val_acc = [0.9113333225250244,
 0.9240000247955322,
 0.9279999732971191,
 0.9413333535194397,
 0.9526666402816772,
 0.940666675567627,
 0.9559999704360962,
 0.9073333144187927,
 0.9573333263397217,
 0.9666666388511658]

acc =[0.8036227226257324,
 0.9014354348182678,
 0.9194121956825256,
 0.9327409267425537,
 0.936978816986084,
 0.9443609118461609,
 0.9502392411231995,
 0.9476418495178223,
 0.9554340243339539,
 0.9555023908615112]
'''