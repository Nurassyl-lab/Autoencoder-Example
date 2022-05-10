import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from skimage import io
import keras
from keras import layers
from numpy.random import permutation
# from keras.datasets import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import random
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow
iters = 20
num_clusters = 9
num_users = 45

h_dim = 783
e_dim = 782

def pre_process(X):
    X = X/255.0
    X = X.reshape((len(X), 784))
    return X

sch = np.linspace(0.1, 0.001, 25)
def scheduler(epoch, lr): 
    lr = sch[epoch - 1]
    return lr

def show_data(X, n=10, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): 
        img = io.imread(os.path.join(folder,filename), as_gray=True)
        if img is not None:
            images.append(img)
    return images

class SERVER:
    def __init__(self):
        self.clusters = []
        self.train_data = []
        self.test_data = []
        self.eval_info = []
        return
    
    def assign_model(self, a, b):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(a, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(a, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(b, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(a, activation = 'relu', name = 'h3'))
        self.model.add(layers.Dense(a, activation = 'relu', name = 'h4'))
        self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001), loss = 'binary_crossentropy')
        return 
    
    def update_ser_weights(self, weights):
        self.model.set_weights(weights)
        return
    def update_server_model(self):
        # this method updates the cluster classification model using the user ones
        w = [clust.model.get_weights() for clust in self.clusters]
        w = np.array(w, dtype=object)
        print('shape is ', w.shape)
        # compute the weight for the update
        l = len(w)
        for i in range(1,l):
            w[0] = w[0] + w[i]
        w[0] = w[0] / float(l)
        print("server model is updated")
        print('w[0] shape is ', w[0].shape)
       
        self.model.set_weights(w[0])
        return 