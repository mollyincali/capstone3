'''
importing images and EDA
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from skimage import io, color, filters
from skimage.transform import resize, rotate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def animal_color_visual(path, animal):
    animal = io.imread(path)
    red = animal[:, :, 0]
    green = animal[:, :, 1]
    blue = animal[:, :, 2]

    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax[0,0].imshow(animal)
    ax[0,0].set_title("Original image")
    ax[0,1].imshow(red, cmap='gray')
    ax[0,1].set_title("Red channel saturation")
    ax[1,0].imshow(blue, cmap='gray')
    ax[1,0].set_title("Blue channel saturation")
    ax[1,1].imshow(green, cmap='gray')
    ax[1,1].set_title("Green channel saturation")
    plt.show();

def show_img(path):
    animal = io.imread(path)
    io.imshow(animal)
    plt.show();
    return animal

    
if __name__ == "__main__":
    #images 512,512,3 dtype unit8
    dog = show_img('../animals/train/dog/flickr_dog_000051.jpg') 

    #tiger change color
    animal_color_visual('../animals/train/wild/flickr_wild_001127.jpg', 'tiger')



#---- GRAPHING CODE TO SAVE 
    #graphs 9 images - not all on 1?!
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #         plt.show();

