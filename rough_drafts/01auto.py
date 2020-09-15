''' 
auto encoders for unsupervised learning on wild folder
code credit
https://github.com/serengil/tensorflow-101/blob/master/python/ConvolutionalAutoencoder.ipynb
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
sns.set()

import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.applications.xception import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten

if __name__ == "__main__":
    img_width, img_height = 150, 150
    train_data_dir = '../animals/train/wild'
    val_data_dir = '../animals/val/wild'
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
        class_mode='input')

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        class_mode = 'input')











# img = imread('../animals/train/wild/flickr_wild_000005.jpg')
# print(type(img))
# plt.imshow(img)
# plt.show();