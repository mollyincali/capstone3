'''
UPDATING 04AUTO
- adding img_gen method
- attempts to get middle layer
- attempts to get KMeans
code credit: https://github.com/alyserecord/wine/blob/master/src/cnn-autoencoder.py
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.cluster import KMeans

class Autoencoder():
    def __init__(self):
        ''' initializes the autoencoder model '''
        self.build_autoencoder()

    def build_autoencoder(self):
        ''' build autoencoder model '''

        autoencoder = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2,2), padding = 'same'),
            layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D((2,2), padding = 'same'),
            layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D((2,2), padding = 'same'),
            layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D((2,2), padding = 'same'),
            
            layers.Flatten(),
            layers.Reshape((8, 8, 8)), #reshape for next layer

            layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same'),
            layers.UpSampling2D((2,2)),
            layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
            layers.UpSampling2D((2,2)),
            layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
            layers.UpSampling2D((2,2)),
            layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
            layers.UpSampling2D((2,2)),
            layers.Conv2D(3, (3,3), activation = 'sigmoid', padding = 'same')
            ])
        
        autoencoder.summary()

        autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

        self.model = autoencoder 

        return self

    def img_gen(self):
        ''' image generator used to reorganize images and pull in using flow from directory '''

        img_width, img_height = 128, 128
        train_data_dir = '../animals/train/w.wild'
        val_data_dir = '../animals/val/w.wild'
        batch_size = 32

        train_datagen = ImageDataGenerator(
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
                shuffle = False,
                class_mode = 'input')

        self.train = train_generator
        self.test = val_generator
        return self.train, self.test

    def fit(self, train, test, batch_size, epochs):
        ''' this method will fit the model using train & test data
        
        input: 
            train = train_generator
            test = validation generator
            batch_size = how many images to feed the model
            epochs = how many rounds of epochs
        '''

        self.model.fit(train, 
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data = test)
        self.history = self.model.history.history

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)
        return self

    def load_weights(self, weights_path):
        ''' load weights of previous model '''
        self.model.load_weights(weights_path)
        return self 

    def get_rmse(self, test):
        ''' calcuate the RMSE of the model after it is trained 

        input: test is array of images from validation generator
        '''

        return self.model.evaluate(test, test)

    def predict(self, X):
        ''' using the trained autoencoder predicts on the provided image array and returns reconstructed images 
        
        input: X is test array of images from validation generator
        output: reconstructed np.array of shape 128,128,3
        ''' 

        return self.model.predict(X)
    
    def get_flat_values(self, X): 
        ''' pull out trained layers of autoencoder
        get the first layer to the flatten layer then predict on the flattened images
        this will be needed for kmeans '''
        encoder = tf.keras.Sequential(self.model.layers[0:9])
        flat_values = encoder.predict(X)
        return flat_values

if __name__ == "__main__":
        auto = Autoencoder()
        auto.build_autoencoder()
        train, test = auto.img_gen()
        auto.fit(train, test, 32, 3)
        flat_values = auto.get_flat_values(train)
        
'''
        #kmeans
        kmeans = KMeans(n_clusters = 5).fit(flat_layers)
        k_labels = kmeans.labels_

        list_filpath = []
        for filename in os.listdir('../animals/train/w.wild/wild'):
                if filename != '.DS_Store':
                        list_filpath.append(f'../animals/train/w.wild/wild/{filename}')
        
        #4738 images
        d = {'cluster': k_labels, 'file_path': list_filpath}
        df = pd.DataFrame(data=d)
'''