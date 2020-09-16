
''' notebook to try to kmeans clustering wild images '''
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
sns.set()
import os
from skimage import color
from skimage import io
from sklearn.cluster import KMeans


def get_val_array(rows):
    array = np.zeros((rows, 262144))
    for idx, filename in enumerate(os.listdir('../animals/val/wild')):
        if filename != '.DS_Store':
            #turns into gray image
            image = color.rgb2gray(io.imread(f'../animals/val/wild/{filename}'))
            #flattens array so shape = 1, 262144
            image = image.reshape(1,512*512) 
            #add row to img_array
            array[idx,:] = image
    return array 

def get_train_array(rows):
    array = np.zeros((rows, 262144))
    for idx, filename in enumerate(os.listdir('../animals/train/wild')):
        if filename != '.DS_Store':
            #turns into gray image
            image = color.rgb2gray(io.imread(f'../animals/train/wild/{filename}'))
            #flattens array so shape = 1, 262144
            image = image.reshape(1, 512*512) 
            #add row to img_array
            array[idx,:] = image
    return array

if __name__ == "__main__":
    val_array = get_val_array(500)
    train_array = get_train_array(4739)

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(train_array)

    from PIL import Image

    from resizeimage import resizeimage

    with open('test-image.jpeg', 'r+b')
        with Image.open() as image:
            is_valid = resizeimage.resize_cover.validate(image, [200, 100])