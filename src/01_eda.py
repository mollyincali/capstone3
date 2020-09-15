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

def pca_work():
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    pca_scaled = scaler.fit_transform(pca_work)
    pca = PCA(n_components=6)
    pca.fit(pca_scaled)
    X_pca = pca.transform(pca_scaled)

    print("\nData after PCA into 6 components")
    print("PC1\tPC2\tPC3\tPC4\tPC5\tPC6")
    for i in range(6):
        print(f'{X_pca[i,0]:0.1f}, \t{X_pca[i,1]:0.1f}, \t{X_pca[i,2]:0.1f}, 
                \t{X_pca[i,3]:0.1f}, \t{X_pca[i,4]:0.1f}, \t{X_pca[i,5]:0.1f}')

    #--- looking for ideal PCA number
    ratio = []
    for num in range(6, 134):
        pca = PCA(n_components=num)
        pca.fit(pca_scaled)
        X_pca = pca.transform(pca_scaled)
        ratio.append([num, np.sum(pca.explained_variance_ratio_)])
    

if __name__ == "__main__":
    dog = show_img('../animals/train/dog/flickr_dog_000051.jpg') #images 512,512,3 dtype unit8

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

