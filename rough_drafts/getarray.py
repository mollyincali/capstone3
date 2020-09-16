''' notebook to get numpy arrays of wild images '''
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
    pass 