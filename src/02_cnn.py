''' class for CNN model '''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import graphing

class CNN():
    def __init__(self, model=None):
        ''' initializes the CNN model '''
        if model != None:
            self.model = model
        else:
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
                    metrics = ['accuracy'])
        
        self.model = cnn
    
    def create_img_gen(self, img_size, batch_size):
        ''' create image generators

        img_size: tuple (width, height)
        batch_size: float 
        
        output: train_generator, val_generator
        '''
        train_data_dir = '../animals/train'
        val_data_dir = '../animals/val'
        batch_size = batch_size

        train_datagen = ImageDataGenerator(
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
            batch_size = 1500,
            shuffle = False,
            class_mode = 'categorical')

        self.train_generator = train_generator
        self.val_generator = val_generator

    def fit_cnn(self, epoch):
        ''' fit the cnn model with modelcheckpoint as a callback '''
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = 'best_mod3.hdf5',
            save_best_only = True,
            monitor = 'val_accuracy',
            mode = 'max')

        self.model.fit(self.train_generator,
            validation_data = self.val_generator,
            epochs = epoch,
            callbacks = [checkpoint])

        self.history = self.model.history.history

    def predict(self, X):
        ''' use to predict on batch of image data gen '''
        return self.model.predict(X)

    def predict_one_img(self, path):
        ''' us model to predict on 1 image given that specific path '''
        test_image = image.load_img(path, target_size = (150,150,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        return self.model.predict(test_image)

if __name__ == "__main__":
    # only use below to fit model
    # cnn = CNN()
    # cnn.build_cnn((150,150,3), 3) 
    # cnn.create_img_gen((150,150), 32)    
    # cnn.fit_cnn(20)

    # #once we have the best model path
    cnn = load_model('best_mod3.hdf5')
    cnn = CNN(model = cnn)
    cnn.create_img_gen((150,150), 32)  

    # get images the model guessed incorrectly
    x, y = next(cnn.val_generator)
    difference = np.argmax(cnn.predict(x), axis = 1) != np.argmax(y, axis = 1)
    diff = x[difference]
    np.sum(diff)
    for d in diff:
        img = (d * 255).astype(np.uint8)
        PIL.Image.fromarray(img).show();

    #call heatmap function
    true = cnn.val_generator.classes
    pred = np.argmax(cnn.predict(cnn.val_generator), axis = 1)
    heatmap(true, pred)