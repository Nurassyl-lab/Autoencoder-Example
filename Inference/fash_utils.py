import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from skimage import io
import keras
from keras import layers
from numpy.random import permutation
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.models import Sequential
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import random
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow
from numpy.random import seed
import random as rd
iters = 20
num_clusters = 9
num_users = 45

h_dim = 392
e_dim = 196

dic = {0:'sweater', 1:'sandals', 2:'boots', 3:'bags', 4:'dress', 5:'pants', 6:'sneakers', 7:'T-shirt', 8:'shirt'}


def custom_loss(y_true, y_pred):
    return y_pred

def mean(vects):
    a, b, c, d, e = vects
    return (a+b+c+d+e)/5.0


def mean_output_shape(shapes):
    shape1, shape2, shape3, shape4, shape5 = shapes
    return shape1

def pre_process(X):
    # X = X/255.0
    X = X.reshape((len(X), 784))
    X = X[permutation(len(X))]
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
    return

def show_one_data(X, n=1, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    plt.imshow(X.reshape((height,width)))
    plt.gray()
    plt.suptitle(title, fontsize = 20)
    return

def save_one_image(X, height = 28, width = 28, path = ""):
    plt.imsave(path, X.reshape((height,width)))
    return



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): 
        img = io.imread(os.path.join(folder,filename), as_gray=True)
        if img is not None:
            images.append(img)
    return images



class USER:
    def __init__(self, i):
        self.name = i
        self.train_data = []
        self.test_data = []
        self.eval_info = []
        self.model = None
        return
    
    def siamese_model(self):
        self.input_1 = layers.Input(shape=(784,), name = 'input_1')
        self.input_2 = layers.Input(shape=(784,), name = 'input_2')
        self.input_3 = layers.Input(shape=(784,), name = 'input_3')
        self.input_4 = layers.Input(shape=(784,), name = 'input_4')
        self.input_5 = layers.Input(shape=(784,), name = 'input_5')
        
        'hidden layer 1'
        self.encoded_1 = layers.Dense(h_dim, activation = 'relu', name = 'in1_h1')(self.input_1)
        self.encoded_2 = layers.Dense(h_dim, activation = 'relu', name = 'in2_h1')(self.input_2)
        self.encoded_3 = layers.Dense(h_dim, activation = 'relu', name = 'in3_h1')(self.input_3)
        self.encoded_4 = layers.Dense(h_dim, activation = 'relu', name = 'in4_h1')(self.input_4)
        self.encoded_5 = layers.Dense(h_dim, activation = 'relu', name = 'in5_h1')(self.input_5)
        
        'encoded layer'
        self.encoded_1 = layers.Dense(e_dim, activation = 'relu', name = 'in1_enc')(self.encoded_1)
        self.encoded_2 = layers.Dense(e_dim, activation = 'relu', name = 'in2_enc')(self.encoded_2)
        self.encoded_3 = layers.Dense(e_dim, activation = 'relu', name = 'in3_enc')(self.encoded_3)
        self.encoded_4 = layers.Dense(e_dim, activation = 'relu', name = 'in4_enc')(self.encoded_4)
        self.encoded_5 = layers.Dense(e_dim, activation = 'relu', name = 'in5_enc')(self.encoded_5)
        
        self.dec = layers.Lambda(mean, output_shape=mean_output_shape)([self.encoded_1, self.encoded_2, self.encoded_3, self.encoded_4, self.encoded_5])
        
        self.model_enc = Model([self.input_1, self.input_2, self.input_3, self.input_4, self.input_5], self.dec)
        self.model_enc.compile(loss=custom_loss, optimizer = 'adam')
        
    def define_model(self):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='softmax', name = 'out'))
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
        return 
    
    def train(self, ep = 1, bc = 16):
        self.model.fit(self.train_data, self.train_data,
                        epochs = ep,
                        batch_size=bc,
                        shuffle=True,
                        validation_data=(self.test_data, self.test_data))
        return
    def train_siamese(self, ep = 20, bc = 16):
        self.model.fit(self.train_data, self.train_data,
                        epochs = ep,
                        batch_size=bc,
                        shuffle=True)
        return

class CLUSTER:
    def __init__(self, number):
        self.users = []
        self.number = number
        self.train_data = []
        self.test_data = []
        self.evs = {}
        self.siam_evs = []
        self.acc = {}
        self.model = None
        self.encs = [[] for i in range(num_clusters)]
        self.argmax = []
        # self.genie = []
        self.hist = []
        self.soft = []
        return

    def last_acc(self, X, title = ''):
        self.acc.update({title : X})
        return
    
    def def_model(self):
        self.input = layers.Input(shape=(784), name = 'input')
        self.x = layers.Dense(h_dim, activation = 'relu', name = 'h1_enc', trainable=True)(self.input)
        self.x = layers.Dense(e_dim, activation = 'relu', name = 'encoded', trainable=True)(self.x)    
        self.x = layers.Dense(h_dim, activation = 'relu', name = 'h2_dec')(self.x)
        self.output = layers.Dense(784, activation='softmax', name = 'out')(self.x)
        self.model = Model(self.input, self.output)
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
    
    def siamese_model(self):
        self.input_1 = layers.Input(shape=(784,), name = 'input_1')
        self.input_2 = layers.Input(shape=(784,), name = 'input_2')
        self.input_3 = layers.Input(shape=(784,), name = 'input_3')
        self.input_4 = layers.Input(shape=(784,), name = 'input_4')
        self.input_5 = layers.Input(shape=(784,), name = 'input_5')
        
        'hidden layer 1'
        self.encoded_1 = layers.Dense(h_dim, activation = 'relu', name = 'in1_h1')(self.input_1)
        self.encoded_2 = layers.Dense(h_dim, activation = 'relu', name = 'in2_h1')(self.input_2)
        self.encoded_3 = layers.Dense(h_dim, activation = 'relu', name = 'in3_h1')(self.input_3)
        self.encoded_4 = layers.Dense(h_dim, activation = 'relu', name = 'in4_h1')(self.input_4)
        self.encoded_5 = layers.Dense(h_dim, activation = 'relu', name = 'in5_h1')(self.input_5)
        
        'encoded layer'
        self.encoded_1 = layers.Dense(e_dim, activation = 'relu', name = 'in1_enc')(self.encoded_1)
        self.encoded_2 = layers.Dense(e_dim, activation = 'relu', name = 'in2_enc')(self.encoded_2)
        self.encoded_3 = layers.Dense(e_dim, activation = 'relu', name = 'in3_enc')(self.encoded_3)
        self.encoded_4 = layers.Dense(e_dim, activation = 'relu', name = 'in4_enc')(self.encoded_4)
        self.encoded_5 = layers.Dense(e_dim, activation = 'relu', name = 'in5_enc')(self.encoded_5)
        
        self.dec = layers.Lambda(mean, output_shape=mean_output_shape, name='mean')([self.encoded_1, self.encoded_2, self.encoded_3, self.encoded_4, self.encoded_5])
        
        self.model_enc = Model([self.input_1, self.input_2, self.input_3, self.input_4, self.input_5], self.dec)
        self.model_enc.compile(loss=custom_loss, optimizer = 'adam')
    
        'adding encoder to decoder model'
        self.model_dec_input = layers.Input(shape=(196,), name = 'dec')
        'hidden layer 2'
        self.model_dec = layers.Dense(h_dim, activation = 'relu', name = 'dec_h2')(self.model_dec_input)
        
        'output layer'
        self.output = layers.Dense(784, activation='sigmoid', name = 'out')(self.model_dec)
        
        'model itself'
        self.model = Model(self.model_dec_input, self.output)
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
        
    def define_model(self):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='softmax', name = 'out'))
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
        return 
    
    def train_siamese(self, enc_ep, enc_bs, dec_ep, dec_bs):
        self.model_enc.fit([self.users[0].train_data, 
                            self.users[1].train_data, 
                            self.users[2].train_data, 
                            self.users[3].train_data, 
                            self.users[4].train_data],
                            self.users[0].train_data,
                           epochs = enc_ep,
                           batch_size = enc_bs)
        
        enc = self.model_enc.predict([self.users[0].test_data, 
                                      self.users[1].test_data, 
                                      self.users[2].test_data, 
                                      self.users[3].test_data, 
                                      self.users[4].test_data])
        
        for i in range(50):
            # n = rd.randint(0, (num_users / num_clusters)-1)
            self.model.fit(enc, self.users[0].train_data, epochs = 1, batch_size = dec_bs)
            self.siam_evs.append(self.model.evaluate(enc, self.users[0].test_data))
        
        
    
    def siam_model_enc(self):
        self.model_enc = Sequential()
        self.model_enc.add(keras.Input(shape=(784,)))
        self.model_enc.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model_enc.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        
    def siam_model_dec(self):
        self.model_dec = Sequential()
        self.model_dec.add(keras.Input(shape=(784,)))
        self.model_dec.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model_dec.add(layers.Dense(784, activation='sigmoid', name = 'out'))
    
    def user_dist(self, n, siamese):
        size = int(num_users / num_clusters)
        
        for i in range(size):
            self.users.append(USER(i))
            if siamese:
                self.users[i].siamese_model()
            else:
                self.users[i].define_model()
        print('These users were assigned to a cluster', self.number)
        for user in self.users:
            print(user.name, end = ' ')
        print('\n\n')
        return
    
    def assign_du(self):
        train_data = self.train_data[permutation(len(self.train_data))]
        test_data = self.test_data[permutation(len(self.test_data))]
        size_test, size_train = int(test_data.shape[0] / num_users), int(train_data.shape[0] / num_users)
        for i in range(len(self.users)):
            self.users[i].train_data = train_data[i*size_train : (i + 1)*size_train]
            self.users[i].test_data = test_data[i*size_test : (i + 1)*size_test]
        return        
    
    def evaluation(self, server):
        print('cluster' + str(self.number)+' evaluation')
        if len(self.evs) == 0:
            for clust in server.clusters:
                self.evs.update({'m_'+str(self.number)+'d_'+str(clust.number): [self.model.evaluate(clust.test_data, clust.test_data)]})
                # if self.number == clust.number:
                #     self.genie += [self.evs['m_'+str(self.number)+'d_'+str(clust.number)]]
        else:
            for clust in server.clusters:
                self.evs['m_'+str(self.number)+'d_'+str(clust.number)] = self.evs['m_'+str(self.number)+'d_'+str(clust.number)] + [self.model.evaluate(clust.test_data, clust.test_data)]
        return
        
    def update_cluster_model(self):
        w = [user.model.get_weights() for user in self.users]
        w = np.array(w, dtype=object)
        l = len(w)
        for i in range(1,l):
            w[0] = w[0] + w[i]
        w[0] = w[0] / float(l)
        self.model.set_weights(w[0])
        return
    
    def graph(self, path, f = None, t = 'loss'):
        plt.figure(figsize=(8, 6))
        plt.subplot(111)
        for e in self.evs:
            if e[2] == e[5]:
                plt.plot(range(len(self.evs[e])), self.evs[e], label = 'm'+dic[int(e[2])]+'_d'+dic[int(e[5])], color = 'black', marker = '.')
                self.last_acc(self.evs[e][-1], e)
            else:  
                plt.plot(range(len(self.evs[e])), self.evs[e], label = 'm'+dic[int(e[2])]+'_d'+dic[int(e[5])])
    
            if f is not None: f.write(e + ',' +str(self.evs[e][-1]) + str('\n'))
        box = plt.subplot(111).get_position()
        plt.subplot(111).set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.xlabel('epochs')
        plt.ylabel('BCL')
        plt.title('cluster '+str(self.number)+' loss')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(path+'c _'+str(self.number)+'_'+t+'.png')
        plt.close()
        show_data(self.model.predict(self.test_data), title = 'cluster ' +str(self.number)+ ' decoded data')
        plt.savefig(path+'clust_'+str(self.number)+'_rec.png')
        plt.close()
        

class SERVER:
    def __init__(self):
        self.clusters = []
        self.train_data = np.array([])
        self.test_data = np.array([])
        self.eval_info = []
        self.evs = []
        self.model = None
        return
    
    def def_model(self):
        self.input = layers.Input(shape=(784), name = 'input')
        self.x = layers.Dense(h_dim, activation = 'relu', name = 'h1_enc', trainable=True)(self.input)
        self.x = layers.Dense(e_dim, activation = 'relu', name = 'encoded', trainable=True)(self.x)    
        self.x = layers.Dense(h_dim, activation = 'relu', name = 'h2_dec')(self.x)
        self.output = layers.Dense(784, activation='softmax', name = 'out')(self.x)
        self.model = Model(self.input, self.output)
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
    
    
    def define_model(self):
        self.model = Sequential()
        self.model.add(keras.Input(shape=(784,)))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h1'))
        self.model.add(layers.Dense(e_dim, activation = 'relu', name = 'encoded'))
        self.model.add(layers.Dense(h_dim, activation = 'relu', name = 'h2'))
        self.model.add(layers.Dense(784, activation='sigmoid', name = 'out'))
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy')
        return 
    
    def clust_dist(self, n, siamese):
        for i in range(n):
            self.clusters.append(CLUSTER(i))
            if siamese:
                self.clusters[i].siamese_model()
            else:
                self.clusters[i].define_model()
        print('These clusters were assigned to server:')
        for clust in self.clusters:
            print(clust.number, end = ' ')
        print('\n\n')
        return
    
    def assign_dc(self):
        size_test, size_train = int(self.test_data.shape[0] / num_clusters), int(self.train_data.shape[0] / num_clusters)
        for clust in self.clusters:
            clust.train_data = self.train_data[clust.number*size_train : (clust.number + 1)*size_train]
            clust.test_data = self.test_data[clust.number*size_test : (clust.number + 1)*size_test]
        return
    
    def update_set_weights(self, weights):
        self.model.set_weights(weights)
        return
    
    def update_server_model(self):
        w = [clust.model.get_weights() for clust in self.clusters]
        w = np.array(w, dtype=object)
        l = len(w)
        for i in range(1,l):
            w[0] = w[0] + w[i]
        w[0] = w[0] / float(l)
        self.model.set_weights(w[0])
        return 
    
    def evalution(self):
        print('server evaluation')
        self.evs = self.evs + [self.model.evaluate(self.test_data, self.test_data)]

    def graph(self, path, t = 'loss'):
        self.test_data = self.test_data[permutation(len(self.test_data))]
        plt.figure()
        plt.plot(range(len(self.evs)), self.evs)
        plt.xlabel('epochs')
        plt.ylabel('BCL')
        plt.title('Server ' + t)
        plt.savefig(path+'server_'+t+'.png')
        plt.close()
        show_data(self.model.predict(self.test_data), title = 'server decoded data')
        plt.savefig(path+'server_rec.png')
        plt.close()
        
    def load_data(self, clust):
        if self.train_data.shape[0] == 0:
            self.train_data = clust.train_data
            self.test_data = clust.test_data
        else:
            self.train_data = np.vstack((self.train_data, clust.train_data))
            self.test_data = np.vstack((self.train_data, clust.test_data))
        
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
