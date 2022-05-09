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
iters = 20
num_clusters = 9
num_users = 45

h_dim = 783
e_dim = 782

def pre_process(X):
    X = X/255.0
    X = X.reshape((len(X), 784))
    return X

def show_data(X, n=10, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)

# def show_data(image):
#     n = 10
#     plt.figure(figsize=(20, 4))
#     for i in range(1, n + 1):
#         ax = plt.subplot(1, n, i)
#         plt.imshow(image[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#     return

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
        self.model.add(layers.Dense(b, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(a, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        return 
    
    def define_model(self):
        # #encoder
        # self.input_img = keras.Input(shape=(28, 28, 1))
        # self.x = layers.Conv2D(32, 3, (2, 2), activation='relu')(self.input_img)
        # self.x = layers.Conv2D(64, 3, (2, 2), activation='relu')(self.x)
        # self.x = layers.Flatten()(self.x)
        # self.x = layers.Dense(units=64)(self.x)
        
        # #decoder
        # self.x = layers.Dense(units=7*7*64, activation="relu")(self.x)
        # self.x = layers.Reshape(target_shape = (7,7,64))(self.x)
        # self.x = layers.Conv2DTranspose(64, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.x = layers.Conv2DTranspose(32, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.decoded = layers.Conv2DTranspose(1, 3, (1, 1), padding = 'SAME', activation='sigmoid')(self.x)
        # self.model = keras.Model(self.input_img, self.decoded)
        # self.model.compile(optimizer='adam', loss='binary_crossentropy')
        # self.model = Sequential()
        input_img = keras.Input(shape=(784,))
        x = layers.Dense(h_dim, activation = 'relu', name = 'h1')(input_img)
        x = layers.Dense(e_dim, activation = 'relu', name = 'encoded')(x)
        x = layers.Dense(h_dim, activation = 'relu', name = 'h2')(x)
        x = layers.Dense(784, activation='sigmoid', name = 'out')(x)
        self.model = Model(input_img, x)
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        # self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        # self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        # self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        # self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        # self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        # self.model.summary()
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
        # fracs = [len(clust.train_data) for clust in self.clusters]
        # tot_data = sum(fracs)
        # fracs = [f/tot_data for f in fracs]
        
        # resulting_weights = self.model.get_weights()
        # for layer in range(len(resulting_weights)):
        #     resulting_weights[layer] = np.array(sum([w[i][layer]*fracs[i] for i in range(len(self.clusters))])) # fed avg
        # self.model.set_weights(resulting_weights)
        # return  
    "9s 31ms/step - loss: 0.6815"
class CLUSTER:
    def __init__(self, number):
        self.users = []
        self.number = number
        self.train_data = []
        self.test_data = []
        self.eval_own_info = np.array([0 for i in range(iters)])
        self.eval_diff_info = np.array([0 for i in range(iters)])
        return
    
    def define_model(self):
        # self.input_img = keras.Input(shape=(28, 28, 1))
        # self.x = layers.Conv2D(32, 3, (2, 2), activation='relu')(self.input_img)
        # self.x = layers.Conv2D(64, 3, (2, 2), activation='relu')(self.x)
        # self.x = layers.Flatten()(self.x)
        # self.x = layers.Dense(units=64)(self.x)
        # self.x = layers.Dense(units=7*7*64, activation="relu")(self.x)
        # self.x = layers.Reshape(target_shape = (7,7,64))(self.x)
        # self.x = layers.Conv2DTranspose(64, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.x = layers.Conv2DTranspose(32, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.decoded = layers.Conv2DTranspose(1, 3, (1, 1), padding = 'SAME', activation='sigmoid')(self.x)
        # self.model = keras.Model(self.input_img, self.decoded)
        # self.model.compile(optimizer='adam', loss='binary_crossentropy')
        # self.input = keras.Input(shape=(784,))
        # self.x = layers.Dense(256, acativation='relu', name = 'h1')(self.input)
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        # self.model.summary()
        return 
    
    def evaluate_own_acc(self):
        a = self.model.evaluate(self.test_data, self.test_data)
        return [a]
        # i = random.randint(0, len(self.test_data) - 4)
        # return self.model.evaluate(self.test_data[i:i+3], self.test_data[i:i+3])

    def evaluate_diff_acc(self, clust):
        a = self.model.evaluate(clust.test_data, clust.test_data)
        return [a]
        # i = random.randint(0, len(clust.test_data) - 4)
        # return self.model.evaluate(clust.test_data[i:i+3], clust.test_data[i:i+3])

    def update_cluster_model(self):
        # this method updates the cluster classification model using the user ones
        w = [user.model.get_weights() for user in self.users]
        w = np.array(w, dtype=object)
        l = len(w)
        print('shape of a clust is ', w.shape)
        print('length is ', l)
        for i in range(1,l):
            w[0] = w[0] + w[i]
        w[0] = w[0] / float(l)
        print('w[0] shape is ', w[0].shape)
        self.model.set_weights(w[0])
        return
        # compute the weight for the update
        # fracs = [len(user.train_data) for user in self.users]
        # tot_data = sum(fracs)
        # fracs = [f/tot_data for f in fracs]
        
        # resulting_weights = self.model.get_weights()
        # for layer in range(len(resulting_weights)):
        #     resulting_weights[layer] = np.array(sum([w[i][layer]*fracs[i] for i in range(len(self.users))])) # fed avg
        # self.model.set_weights(resulting_weights)
        # return  

class USER:
    def __init__(self, i):
        self.name = i
        self.train_data = []
        self.test_data = []
        self.eval_info = []
        return
        
    def define_model(self):
        # self.input_img = keras.Input(shape=(28, 28, 1))
        # self.x = layers.Conv2D(32, 3, (2, 2), activation='relu')(self.input_img)
        # self.x = layers.Conv2D(64, 3, (2, 2), activation='relu')(self.x)
        # self.x = layers.Flatten()(self.x)
        # self.x = layers.Dense(units=64)(self.x)
        
        
        # self.x = layers.Dense(units=7*7*64, activation="relu")(self.x)
        # self.x = layers.Reshape(target_shape = (7,7,64))(self.x)
        # self.x = layers.Conv2DTranspose(64, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.x = layers.Conv2DTranspose(32, 3, (2, 2), padding = 'SAME', activation='relu')(self.x)
        # self.decoded = layers.Conv2DTranspose(1, 3, (1, 1), padding = 'SAME', activation='sigmoid')(self.x)
        # self.model = keras.Model(self.input_img, self.decoded)
        # self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        return 
    
    def train(self, ep):
        self.model.fit(self.train_data, self.train_data,
                        epochs = ep,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(self.test_data, self.test_data))
        # self.model.summary()
        return  
    
def plot_two_accs(clust, t = ''):
    plt.figure()
    plt.title(t)
    plt.plot(range(20), clust.eval_own_acc[0:20], label = 'own data')
    plt.plot(range(20), clust.eval_diff_acc[0:20], label = 'diff data')
    plt.legend()
    plt.savefig(t+'.png')
    return
    
def clean_own(clust):
    clust.eval_own_acc = np.array([0.0 for i in range(20)])
    return

def clean_diff(clust):
    clust.eval_diff_acc = np.array([0.0 for i in range(20)])    
    return

#==============================================================================
test_small_data = []
train_small_data = []
for i in range(9):
    test = np.array(load_images_from_folder('federated_learning/federated_fashion_mnist/federated_fashion_mnist_80/'+str(i)+'/test_images/'))
    train = np.array(load_images_from_folder('federated_learning/federated_fashion_mnist/federated_fashion_mnist_80/'+str(i)+'/training_images/'))
    test = np.reshape(test, (len(test), 28, 28))
    train = np.reshape(train, (len(train), 28, 28))
    test = pre_process(test)
    train = pre_process(train)
    test_small_data.append(test)
    train_small_data.append(train)

data_test = []
data_train = []
path = 'federated_learning/federated_fashion_mnist/federated_fashion_mnist_80/'
for i in range(9):
    data_test.append(load_images_from_folder(path+str(i)+'/test_images/'))
    data_train.append(load_images_from_folder(path+str(i)+'/training_images/'))

data_test = np.array(data_test)
data_train = np.array(data_train)

data_test = data_test.reshape(8991, 28, 28)
data_train = data_train.reshape(8991, 28, 28)

data_test = pre_process(data_test)
data_train = pre_process(data_train)

shuffler1 = permutation(data_test.shape[0]) # from numpy
shuffler2 = permutation(data_train.shape[0]) # from numpy
data_test = data_test[shuffler1]
data_train = data_train[shuffler2]

(fm_train, _), (fm_test, _) = fashion_mnist.load_data()
fm_train = pre_process(fm_train)
fm_test = pre_process(fm_test)
#==============================================================================

# def update_cluster_model_weigths(a, b):
#     # this method updates the cluster classification model using the user ones
#     # w = [user.get_model().get_weights() for user in a.users]
#     w = a.model.get_weights()
#     # compute the weight for the update
#     fracs = [len(user.data['labels']) for user in a.users]
#     fracs = [len(a.train_data)]

#     tot_data = sum(fracs)
#     fracs = [f/tot_data for f in fracs]

#     resulting_weights = b.model.get_weights()
#     for layer in range(len(resulting_weights)):
#         resulting_weights[layer] = np.array(sum(w[layer]*fracs[i])) # fed avg
#     b.model.set_weights(resulting_weights)
#     return  

