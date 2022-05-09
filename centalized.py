from fash_utils import *
import math
from numpy.random import seed

import tensorflow

matplotlib.use('Agg')
# matplotlib.use('qt5agg')
"federated fashion mnist"
main_server = SERVER()
(main_server.train_data, _), (main_server.test_data, _) = fashion_mnist.load_data()
main_server.train_data=pre_process(main_server.train_data)
main_server.test_data=pre_process(main_server.test_data)

h_dim = [780, 700, 500, 450, 450, 450, 392, 256, 98, 26, 400, 400, 400, 26]
e_dim = [729, 576, 400, 196, 100, 49, 196, 100, 49, 16, 64, 36, 10, 10]
S = np.arange(1, 11)
path = 'centralized/'
lis = {}
    
for s in S:
    seed(s)
    tensorflow.random.set_seed(s)
    for a, b in zip(h_dim, e_dim):
        main_server.assign_model(a = a, b = b)
        history = main_server.model.fit(main_server.train_data, main_server.train_data, epochs = 50, batch_size=64)
        if s == 1:
            plt.figure()
            plt.title('4 dense layers | LOSS | h_dim = ' + str(a) + ' e_dim = ' + str(b))
            plt.plot(range(len(history.history['loss'])), history.history['loss'])
            plt.xlabel('epochs')
            plt.ylabel('binary crossentropy loss')
            plt.savefig(path+str(a)+'_'+str(b)+'.png')
            plt.close()
        # show_data(main_server.model.predict(main_server.test_data), title='decoded data')
        # plt.savefig(path+str(a)+'_'+str(b)+'rec'+'.png')
        # plt.close()
        # get_encoded_data = Model(inputs=main_server.model.input, outputs=main_server.model.get_layer("encoded").output)
        # encoded_data = get_encoded_data.predict(main_server.test_data)
        # show_data(encoded_data, height=int(math.sqrt(b)), width=int(math.sqrt(b)), title="encoded data")
        # plt.savefig(path+str(a)+'_'+str(b)+'enc'+'.png')
        # plt.close()
        # lis.append(str(a)+'_'+str(b)+'final loss value = '+str(history.history['loss'][-1]))
        if s == 1: 
            lis.update({str(a)+'_'+str(b) : history.history['loss'][-1]})
        else:
            lis[str(a)+'_'+str(b)] = lis[str(a)+'_'+str(b)] + history.history['loss'][-1]
        
        if s == 10:
            lis[str(a)+'_'+str(b)] = lis[str(a)+'_'+str(b)] / 10.0