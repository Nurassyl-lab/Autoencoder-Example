from fash_utils import *
#==============================================================================
"initial parameters"
iterations = [0, 1, 2, 3, 4, 5, 6 ,7 ,8 , 9]
# iterations = []
heterogeneity = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# heterogeneity = [100]

ep = 1
bs = 16
com_rounds = 20

save_models = True#if True, data will be saved
save_curves = False #if true curves won't be displayed but saved
save_model_choosing = False#models that server implements are gonna be written to the txt file
load_models = False#if u dont want to train, the latest models are gonna be load, also it will load test_data
save_decoded = False#if true reconstructed images will be saved using fully trained model
#==============================================================================
for het in heterogeneity:
    img_acc = 0
    pix_acc = 0
    gen_acc = 0
    fed_acc = 0
    
    " *files "
    path_to_save = 'federated_learning/results_het'+str(het)+'/'
    path_to_load = 'federated_learning/federated_fashion_mnist/federated_fashion_mnist_'+str(het)+'/'

    for iteration in iterations:    
        
        clust_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
        seed(iteration)#same seeds will generate same results
        tensorflow.random.set_seed(iteration)
        
        " *create server "
        main_server = SERVER()
        # main_server.model = None #since server will take models of different clusters, assign server model as None
        main_server.def_model()
        
        " *load fashion mnist data "
        clust_data = [[], []]
        for i in range(num_clusters):
            clust_data[0].append(np.array(load_images_from_folder(path_to_load+str(i)+'/training_images/')))
            clust_data[1].append(np.array(load_images_from_folder(path_to_load+str(i)+'/test_images/')))
    
    #==============================================================================
        " *assign clusters to server, and to server "
        
        'assign clusters to a server'
        for i in range(num_clusters):
            main_server.clusters.append(CLUSTER(i))
            main_server.clusters[i].def_model()
        print('These clusters were assigned to server:')
        for clust in main_server.clusters:
            print(clust.number, end = ' ')
        print('\n\n')
        
        " *assign data to clusters "
        for i in range(num_clusters):
            main_server.clusters[i].train_data = pre_process(np.array(clust_data[0][i]))
            main_server.clusters[i].test_data = pre_process(np.array(clust_data[1][i]))
            
        " *combine data from clusters and assign it to server " 
        for clust in main_server.clusters:
            main_server.load_data(clust)
    #==============================================================================
        " *training or loading models "
        test_data = np.array([])#create a smaller version of test data
        
        pixel_loss = []
        image_loss = []
        federated_loss = []
        genie_loss = []
        
        if save_model_choosing:
            f_gen = open(path_to_save+'cluster_choosing_genies_iter'+str(iteration)+'.txt', 'w')
            f_img = open(path_to_save+'cluster_choosing_images_iter'+str(iteration)+'.txt', 'w')
            f_pix = open(path_to_save+'cluster_choosing_pixels_iter'+str(iteration)+'.txt', 'w')
        
        for cr in range(com_rounds):
            print("\n**Com. round = "+str(cr) + " iteration: " + str(iteration))
            # if save_model_choosing:
            #     f_gen.write('For CR ' + str(cr)+'\n')
            #     f_img.write('For CR ' + str(cr)+'\n')
            #     f_pix.write('For CR ' + str(cr)+'\n')
        
            if load_models:
                if com_rounds == 1:
                    cr = 19#load fully trained model
                for clust in main_server.clusters:
                    print("loading data for cluster", clust.number)
                    clust.model = keras.models.load_model(path_to_save+"iteration"+str(iteration)+"_cr"+str(cr)+"_clust"+str(clust.number)+".h5")
                    if clust.number == 0:
                        # test_data = np.array(clust.test_data[0:250])
                        test_data = np.array(clust.test_data)
                    else:
                        # test_data = np.append(test_data, clust.test_data[0:250], axis = 0)
                        test_data = np.append(test_data, clust.test_data, axis = 0)
            else:
                for clust in main_server.clusters:
                    print(clust.number,"cluster is training")
                    clust.model.fit(clust.train_data, clust.train_data, epochs = ep, batch_size = bs, shuffle = True)
                    clust.model.save(path_to_save+"iteration"+str(iteration)+"_cr"+str(cr)+"_clust"+str(clust.number)+"_het"+str(het)+".h5")
                    # clust.model.save(path_to_save+'C'+str(clust.number)+'CR'+str(j)+'.h5')#CR is comm round
                    if clust.number == 0 and cr == 0:
                        # test_data = np.array(clust.test_data[0:250])
                        test_data = np.array(clust.test_data)
                    elif cr == 0:
                        # test_data = np.append(test_data, clust.test_data[0:250], axis = 0)
                        test_data = np.append(test_data, clust.test_data, axis = 0)
    #==============================================================================
        " *inference part "
        #get softmax outputs
        softmax_out = []
        for c in range(num_clusters):
            softmax_out.append(main_server.clusters[c].model.predict(test_data))
            softmax_out[c] = np.abs(0.5-np.array(softmax_out[c]))#since it can't determine whether it is 0 or 1
            if save_decoded:
                matplotlib.use('Agg')
                show_data(softmax_out[c])
                plt.savefig(path_to_save+"reconstructed/"+"clust"+str(c)+"_het"+str(het)+".png")
                plt.close()
                matplotlib.use('qt5agg')

        #these are final results that we are trying to evaluate
        img_loss = 0
        pix_loss = 0
        gen_loss = 0
        fed_loss = 0
        
        clust_data = 0
        genie_count = 0
        for i in range(len(test_data)):#take a single image from dataset
            #since test_data is structurized I can write if else statement as follows
            if i >= clust_data * (test_data.shape[0]/num_clusters) + (test_data.shape[0]/num_clusters):
                clust_data += 1
    
            #genie evaluation per image
            #this part will be modified later, since I want to add classification model to the 
            if i >= (test_data.shape[0]/num_clusters) * clust_data and i < (test_data.shape[0]/num_clusters) * clust_data + ((test_data.shape[0]/num_clusters) * (het/100)):
                gen_loss += main_server.clusters[clust_data].model.evaluate(np.expand_dims(test_data[i], axis=0), np.expand_dims(test_data[i], axis=0))
                genie_count += 1
                if save_model_choosing: f_gen.write('Image_'+str(i)+' genies model is cluster_'+str(clust_data)+'\n')

            #best cluster selection by image evaluation
            args = []
            #find best cluster for single image
            for c in range(num_clusters):
                image = softmax_out[c][i]#image i of cluster c
                args.append(image[np.argmax(image)])
            best_clust = np.argmax(args)
            if save_model_choosing: f_img.write('Image_'+str(i)+' best model selected by server is cluster_'+str(best_clust)+'\n')
            img_loss += main_server.clusters[best_clust].model.evaluate(np.expand_dims(test_data[i], axis=0), np.expand_dims(test_data[i], axis=0))
            
            #best cluster selection by pixel evaluation
            matrix_softs = np.array([softmax_out[0][i]])
            for clust in range(1, num_clusters):
                matrix_softs = np.vstack((matrix_softs, softmax_out[clust][i]))
            for column in range(test_data.shape[1]):
                best_pix = np.argmax(matrix_softs[:,column])
                clust_count[best_pix]+=1
            index = list(dict(sorted(clust_count.items(), key=lambda item: item[1])).items())[-1][0]
            if save_model_choosing: f_pix.write('Image_'+str(i)+' best model selected by server(pixel evaluation) is cluster_'+str(index)+'\n')
            pix_loss += main_server.clusters[index].model.evaluate(np.expand_dims(test_data[i], axis=0), np.expand_dims(test_data[i  ], axis=0))
        
        #append to the final arrays(curves)
        img_loss = img_loss / len(test_data)
        # image_loss.append(img_loss)
        
        pix_loss = pix_loss / len(test_data)
        # pixel_loss.append(pix_loss)
        
        gen_loss = gen_loss / genie_count
        # genie_loss.append(gen_loss)
        
        #combine gradients from clusters(federated average)
        w = [clust.model.get_weights() for clust in main_server.clusters]
        weigths = [[] for _ in range(len(w[0]))]
        for k in range(len(w[0])):#selects weights(w) row
            tmp = []
            for c in range(num_clusters):
                if c == 0:
                    tmp = w[c][k]
                else:
                    tmp = tmp + w[c][k]
            tmp = tmp/num_clusters
            weigths[k] = tmp
        main_server.model.set_weights(weigths)
        # federated_loss.append(main_server.model.evaluate(test_data, test_data))
        fed_loss = main_server.model.evaluate(test_data, test_data)
        
        img_acc += (1 - img_loss)
        pix_acc += (1 - pix_loss)
        gen_acc += (1 - gen_loss)
        fed_acc += (1 - fed_loss)
    img_acc /= len(iterations)
    pix_acc /= len(iterations)
    gen_acc /= len(iterations)
    fed_acc /= len(iterations)
    f_accs = open(path_to_save+'accuracy_het'+str(het)+'.txt', 'w')
    f_accs.write('genie accuracy,'+str(gen_acc)+"\n")
    f_accs.write('imagewise accuracy,'+str(img_acc)+"\n")
    f_accs.write('pixelwise accuracy,'+str(pix_acc)+"\n")
    f_accs.write('federated av. accuracy,'+str(fed_acc)+"\n")
    f_accs.close()
    
    
    
    # img_acc = img_acc + (1 - np.array(image_loss))
    # pix_acc = pix_acc + (1 - np.array(pixel_loss))
    # gen_acc = gen_acc + (1 - np.array(genie_loss))
    # fed_acc = fed_acc + (1 - np.array(federated_loss))
    # "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    # img_acc = img_acc / len(iterations)
    # pix_acc = pix_acc / len(iterations)
    # gen_acc = gen_acc / len(iterations)
    # fed_acc = fed_acc / len(iterations)
    
    # if save_curves: matplotlib.use('Agg')
    # else: matplotlib.use('qt5agg')

    # plt.figure()
    
    # plt.plot(range(len(img_acc)), img_acc, 'b', label = 'softmax by image')
    # plt.plot(range(len(pix_acc)), pix_acc, 'y', label = 'softmax by pixel')
    # plt.plot(range(len(gen_acc)), gen_acc, 'r', label = 'genie')
    # plt.plot(range(len(fed_acc)), fed_acc, 'g', label = 'federated average')
    
    # plt.title('Server Accuracy')
    # plt.xlabel('communicational round')
    # plt.ylabel('Acc = 1 - loss(BCL)')
    # plt.legend()    
    # if save_curves: plt.savefig(path_to_save+"server_accuracy_for_het_"+str(het)+'.png')
    
    # if save_model_choosing:
    #     f_gen.close()
    #     f_img.close()
    #     f_pix.close()