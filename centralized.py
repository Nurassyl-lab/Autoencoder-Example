'NURA'
from fash_utils import *
import math
from numpy.random import seed
import numpy as np

matplotlib.use('Agg')
# matplotlib.use('qt5agg')
"federated fashion mnist"
main_server = SERVER()
(main_server.train_data, _), (main_server.test_data, _) = fashion_mnist.load_data()
main_server.train_data=pre_process(main_server.train_data)
main_server.test_data=pre_process(main_server.test_data)

h_dim = [780, 450, 450, 450, 256, 26]
e_dim = [729, 196, 100, 49, 100, 10]
S = np.arange(1, 6)
ep = 25
bs = 64
path = 'centralized/'
lis = {}
bcl = {}

for s in S:
    seed(s)
    tensorflow.random.set_seed(s)
    for a, b in zip(h_dim, e_dim):
        print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print('s = ', s)
        print('a, b =', a, ',',b)
        main_server.assign_model(a = a, b = b)
        
        # main_server.model.summary()
        
        history = main_server.model.fit(main_server.train_data, main_server.train_data, epochs = ep, batch_size = bs)
        main_server.model.save(path+str(a)+"_"+str(b)+'SLSM'+str(s)+".h5")#a_b(SIX LAYER Server model)seed
        
        
        show_data(main_server.model.predict(main_server.test_data), title='decoded data')
        plt.savefig(path+str(a)+'_'+str(b)+'rec'+'_s'+str(s)+'.png')
        plt.close()
        
        "uncomment to get encoded data pictures"
        "NOTE! that there will be an error for the size 10, so if you want to use below code"
        "remove size 10 from e_dim"
        # get_encoded_data = Model(inputs=main_server.model.input, outputs=main_server.model.get_layer("encoded").output)
        # encoded_data = get_encoded_data.predict(main_server.test_data)
        # show_data(encoded_data, height=int(math.sqrt(b)), width=int(math.sqrt(b)), title="encoded data")
        # plt.savefig(path+str(a)+'_'+str(b)+'enc'+'.png')
        # plt.close()
        # lis.update(str(a)+'_'+str(b)+'final loss value = '+str(history.history['loss'][-1]))
        
        if s == S[0]: 
            bcl.update({str(a)+'_'+str(b) : history.history['loss']})
            lis.update({str(a)+'_'+str(b) : history.history['loss'][-1]})
        else:
            bcl[str(a)+'_'+str(b)] = np.add(bcl[str(a)+'_'+str(b)], history.history['loss'])
            lis[str(a)+'_'+str(b)] = lis[str(a)+'_'+str(b)] + history.history['loss'][-1]
        
        if s == S[-1]:
            plt.figure()
            plt.title('6 dense layers | LOSS | h_dim = ' + str(a) + ' e_dim = ' + str(b))
            bcl[str(a)+'_'+str(b)] = bcl[str(a)+'_'+str(b)] / float(len(S))
            plt.plot(range(len(bcl[str(a)+'_'+str(b)])), bcl[str(a)+'_'+str(b)])
            plt.xlabel('epochs')
            plt.ylabel('binary crossentropy loss')
            plt.savefig(path+str(a)+'_'+str(b)+'.png')
            plt.close()

            lis[str(a)+'_'+str(b)] = lis[str(a)+'_'+str(b)] / float(len(S))