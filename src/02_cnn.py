'''
cnn model for image classification
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

def data_gen():
    img_width, img_height = 150, 150

    data_dir = 'UPDATE'
    validation_data_dir = 'UPDATE'
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batch_size = 16
    
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)