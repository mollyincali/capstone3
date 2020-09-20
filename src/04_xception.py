''' 
transfer learning with xception architecture
predictions from xception will be used to cluster images
'''
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from glob import glob 
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop
from sklearn.cluster import KMeans

def process_img(filename):
    ''' loads one image from filename, preprocesses it and expands the dimensions '''
    original = load_img(filename, target_size = (299,299))
    numpy_image = preprocess_input(img_to_array(original))
    image_batch = np.expand_dims(numpy_image, axis =0)
    return image_batch

def predict_arrays(path):
    ''' imports images, makes prediction using xception model '''
    array = np.zeros((500, 1000))
    file_path = []
    for idx, img in enumerate(glob("../animals/val/wild/*")):
        im = process_img(img)
        file_path.append(img)
        xcept_feature = model.predict(im)
        array[idx,:] = xcept_feature
    return array, file_path

def cluster_images(df):
    ''' displays kmeans cluster of animals '''
    for i in range(5):
        group = df[df['cluster'] == i].copy()
        plt.figure(figsize=(10, 10))
        for idx, img in enumerate(group.sample(9).iloc[:,1]):
            animal = io.imread(img)
            ax = plt.subplot(3, 3, idx + 1)
            plt.imshow(animal)
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    #used these weights to predict my images
    model = Xception(weights='imagenet',
                    include_top=True,
                    input_shape=(299,299,3))

    xcept_array, file_path = predict_arrays("../animals/val/wild/*")
    print("array created")

    # kmeans on xcept array predictions
    kmeans = KMeans(n_clusters = 5).fit(xcept_array)
    k_labels = kmeans.labels_
    print("kmeans complete")

    # get group images
    d = {'cluster': k_labels, 'file_path': file_path}
    df = pd.DataFrame(data=d)

    # display group images
    cluster_images(df)
