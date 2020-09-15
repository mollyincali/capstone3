'''
kmeans cluster on wild images
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.cluster import KMeans
from keras.preprocessing.image import load_img, img_to_array, array_to_img

