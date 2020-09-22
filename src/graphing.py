''' code for graphing used throughout capstone '''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from skimage import io
import PIL 

def cluster_images(df):
    ''' show kmeans cluster of animals '''
    for i in range(5):
        group = df[df['cluster'] == i].copy()
        plt.figure(figsize=(10, 10))
        for idx, img in enumerate(group.sample(9).iloc[:,1]):
            animal = io.imread(f'../oganimals/val/w.wild/{img}')
            ax = plt.subplot(3, 3, idx + 1)
            plt.imshow(animal)
            plt.axis("off")
        plt.show()

def get_before_after(x, decoded):
    ''' show before entering autoencode and after running through autoencoder '''
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def graph_model(history, epochs):
    ''' code to run accuracy on test and validation '''
    pink = '#CC9B89'
    blue = '#23423F'
    # gold = '#B68D34'
    epochs = 20

    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    loss=history.history['loss'] 
    val_loss=history.history['val_loss'] 
    epochs_range = range(epochs)

    if min(val_acc) < min(acc):
        min_ = min(val_acc)
    else:
        min_ = min(acc)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy',
            linewidth = 2, color = blue)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy',
            linewidth = 2, color = pink)
    plt.legend(loc='lower right')
    plt.ylim((min_,1))
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss',
            linewidth = 2, color = blue)
    plt.plot(epochs_range, val_loss, label='Validation Loss',
            linewidth = 2, color = pink)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def heatmap(true, pred):  
    ''' creates heatmap of true and predicited animals '''
    cm = confusion_matrix(true, pred)
    group_names = ['True Cat',' ',' ',
                    ' ','True Dog',' ',
                    ' ',' ','True Wild']
    group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(3,3)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['Cat','Dog','Wild'], yticklabels=['Cat','Dog','Wild'])
    plt.show();

if __name__ == "__main__":
    pass 