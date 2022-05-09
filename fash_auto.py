lis = []
from fash_utils import *
matplotlib.use('Agg')
"federated fashion mnist"
# for C in range(9):    
for C in range(1):
    # C = 0
    # MAIN_eval_own = []
    # MAIN_eval_diff = []
    # MAIN_eval_server = []
    "define server model and data"
    main_server = SERVER()
    main_server.define_model()
    (main_server.train_data, _), (main_server.test_data, _) = fashion_mnist.load_data()
    main_server.train_data=pre_process(main_server.train_data)
    main_server.test_data=pre_process(main_server.test_data)
    # main_server.test_data= data_test
    # main_server.train_data= data_train
    size_test, size_train = int(main_server.test_data.shape[0] / num_clusters), int(main_server.train_data.shape[0] / num_clusters)
    for i in range(num_clusters):
        main_server.clusters.append(CLUSTER(i))
        main_server.clusters[i].define_model()
        main_server.clusters[i].test_data = main_server.test_data[i*size_test : (i+1) * size_test]
        main_server.clusters[i].train_data = main_server.train_data[i*size_train : (i+1) * size_train]
        # main_server.clusters[i].test_data = test_small_data[i]
        # main_server.clusters[i].train_data = train_small_data[i]
        print("*Data for cluster", i, "is ready, and its model is defined")
        
    print("\n========================================================================\n")
    
    u = 0
    for clust in main_server.clusters:
        index = 0
        for i in range(int(num_users/num_clusters)):
            clust.users.append(USER(u))
            clust.users[index].define_model()
            print("user",u,"is assigned to cluster",clust.number,"and its model was defined")
            u+=1
            index+=1
    
    # "prepare data for users"
    # user_train_data = []
    # user_test_data = []
    # for i in range(num_users):
    #     'problem: users always train same pictures'
    #     user_train_data.append(data_train[i*199:(i+1)*199])
    #     user_test_data.append(data_test[i*199:(i+1)*199])
    
    comm_rounds = 20
    
    # for i in range(C):
    evals_own = [[],[],[],[],[],[],[],[],[]]
    evals_diff = [[],[],[],[],[],[],[],[],[]]
    server_own = []
    # C = 0#using cluster 1
    print("\n**MAIN TRAINING HAS STARTED\n")
    # C = 0
    for i in range(comm_rounds):
        print("\n   Communicational round " +str(i)+ "\n")
        for clust in main_server.clusters:
            u = 0
            for users in clust.users:
            # for u in range(9):
                # r = random.randint(0, 44)
                # users.train_data = user_train_data[r]
                # users.test_data = user_test_data[r]
                size_train, size_test = int(clust.train_data.shape[0] / num_users), int(clust.test_data.shape[0] / num_users )
                users.train_data = clust.train_data[size_train*u:(u+1)*size_train]
                # j = random.randint(0, 9)
                # users.train_data = main_server.clusters[j].train_data[199*u:(u+1)*199]
                users.test_data = clust.test_data[size_test*u:(u+1)*size_test]
                # users.train_data = main_server.clusters[j].test_data[199*u:(u+1)*199]
                u+=1
                users.train(1)
            clust.update_cluster_model()
            
            print("   Evaluation started")
            # for i in range(10):
            evals_own[clust.number].append(clust.evaluate_own_acc())#genie
            evals_diff[clust.number].append(clust.evaluate_diff_acc(main_server.clusters[C]))
            # evals_own = evals_own/10
            # evals_diff = evals_diff/10
            print("\n\nShape is",np.array(evals_own).shape, np.array(evals_diff).shape)
        # if C == 0:
        main_server.update_server_model()
        server_own.append(main_server.model.evaluate(main_server.test_data, main_server.test_data))
                
    # MAIN_eval_own.append(evals_own)
    # MAIN_eval_diff.append(evals_diff)
    # MAIN_eval_server.append(server_own) 
    
    # print("MAIN SHAPE IS ", np.array(MAIN_eval_own).shape)
    "loss mse accuracy"
    a = []
    b = []
    acc_own = [[],[],[],[],[],[],[],[],[]]
    acc_diff = [[],[],[],[],[],[],[],[],[]]
    for c in range(num_clusters):
        tmp1 = []
        tmp2 = []
        for i in range(len(evals_own[c])):
            # print(evals_own[0][i][0])
            tmp1.append(evals_own[c][i][0])
            tmp2.append(evals_diff[c][i][0])
            a.append(1.0 - evals_own[c][i][0])
            b.append(1.0 - evals_diff[c][i][0])
        plt.figure(main_server.clusters[c].number)
        plt.title('LOSS : cluster '+str(main_server.clusters[c].number))
        plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
        plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster' +str(C)+ '| jacket')
        plt.legend()
        plt.savefig(str(C)+'clust_loss'+str(str(main_server.clusters[c].number))+".png")
        plt.close()  
        acc_own[c].append(a)
        acc_diff[c].append(b)
    "loss mse accuracy"
    
    for c in range(num_clusters):
        tmp1 = []
        tmp2 = []
        for i in range(len(evals_own[c])):
            # print(evals_own[0][i][0])
            # tmp1.append(acc_own[c][0][i])
            # tmp2.append(acc_diff[c][0][i])
            tmp1.append(1 - evals_own[c][i][0])
            tmp2.append(1 - evals_diff[c][i][0])
        plt.figure(main_server.clusters[c].number)
        plt.title('ACC : cluster '+str(main_server.clusters[c].number))
        plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
        plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster' +str(C)+ '| jacket')
        plt.legend()
        plt.savefig(str(C)+'clust_acc'+str(str(main_server.clusters[c].number))+".png")
        plt.close()  
        # print("cluster"+ str(c),tmp2[-1], " data of cluster " + str(C))
        lis.append("cluster" + str(c) + " acc is " + str(tmp2[-1]) + ", data of cluster " + str(C))    
    
matplotlib.use('qt5agg')
show_data(main_server.clusters[0].users[0].model.predict(main_server.clusters[0].users[0].test_data))

# show_data(main_server.clusters[0].users[0].model.predict(main_server.clusters[0].test_data))

show_data(main_server.clusters[0].model.predict(main_server.clusters[0].test_data))

show_data(main_server.clusters[0].users[0].test_data)

for c in range():
    tmp1 = []
    tmp2 = []
    for i in range(len(evals_own[c])):
        # print(evals_own[0][i][0])
        tmp1.append(MAIN_eval_own[0][c][i][1])
        tmp2.append(MAIN_eval_diff[0][c][i][1])
    plt.figure(main_server.clusters[c].number)
    plt.title('MSE : cluster '+str(main_server.clusters[c].number))
    plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
    plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster ' +str(C)+ ' | jacket')
    plt.legend()
    plt.savefig(str(C)+'clust_mse'+str(str(main_server.clusters[c].number))+".png")
    plt.close()   
"loss mse accuracy"
for c in range(9):
    tmp1 = []
    tmp2 = []
    for i in range(len(evals_own[c])):
        # print(evals_own[0][i][0])
        tmp1.append(MAIN_eval_own[0][c][i][2])
        tmp2.append(MAIN_eval_diff[0][c][i][2])
    plt.figure(main_server.clusters[c].number)
    plt.title('OVERALL ACCURACY : cluster '+str(main_server.clusters[c].number))
    plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
    plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster ' +str(C)+ ' | jacket')
    plt.legend()
    plt.savefig(str(C)+'clust_acc'+str(str(main_server.clusters[c].number))+".png")
    plt.close()  
    





acc = []
loss = []
mse = []
for i in range(len(MAIN_eval_server[0])):
    loss.append(MAIN_eval_server[0][i][0])
    mse.append(MAIN_eval_server[0][i][1])
    acc.append(MAIN_eval_server[0][i][2])

plt.figure()
plt.title('LOSS of server')
plt.plot(range(len(loss)), loss, label = 'server loss')
plt.legend()
plt.savefig('server_LOSS.png')
plt.close()   

plt.figure()
plt.title('MSE of server')
plt.plot(range(len(mse)), mse, label = 'server mse')
plt.legend()
plt.savefig('server_MSE.png')
plt.close()   
    
plt.figure()
plt.title('ACC of server')
plt.plot(range(len(acc)), acc, label = 'server acc')
plt.legend()
plt.savefig('server_ACC.png')            
plt.close()    
            "asdasddddddddddddddddddd"
    # #save_models
    # path = 'federated_learning/fash_models_1/'
    # main_server.model.save(path+'server.h5')
    # for clust in main_server.clusters:
    #     clust.model.save(path+"cluster"+str(clust.number)+".h5")
    #     for user in clust.users:
    #         user.model.save(path+"user"+str(user.name)+".h5")
    
    # #save evalautions
    # with open(path+'evaluations.txt', 'w') as f:
    #     for x,y in zip(evals_own, evals_diff):
    #         f.write("new cluster")
    #         for a, b in zip(x , y):
    #             f.write("%s,%s\n" %(a,b))
                
                
    # def plot(x, title = ''):
    #     plt.figure()
    #     plt.plot(range(len(x)), x)
    #     plt.title(title)
    #     return
                
    # evs = []            
    # from fash_utils import *   
    # user = USER(1)
    # user.train_data = user_train_data[0]
    # user.test_data = user_test_data[0]
    # user.define_model()
    
    # # user.train(50)
    
    # history = user.model.fit(train_small_data[0], train_small_data[0],
    #                 epochs = 50,
    #                 batch_size=1,
    #                 shuffle=True,
    #                 validation_data=(test_small_data[0], test_small_data[0]))
    
    # # show_data(user.model.predict(user.test_data))
    # show_data(user.model.predict(test_small_data[0]))
    # show_data(test_small_data[0])
    # # evs.append(user.model.evaluate(user.test_data, user.test_data))
    
    # plot(history.history['loss'], title='loss')
    
    # plot(history.history['mse'], title='mse')
    
    # plot(history.history['accuracy'], title='accuracy')
    
    
    
    "loss mse accuracy"
    for c in range(num_clusters):
        tmp1 = []
        tmp2 = []
        for i in range(len(evals_own[c])):
            # print(evals_own[0][i][0])
            tmp1.append(evals_own[c][i][0])
            tmp2.append(evals_diff[c][i][0])
        plt.figure(main_server.clusters[c].number)
        plt.title('LOSS : cluster '+str(main_server.clusters[c].number))
        plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
        plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster' +str(C)+ '| jacket')
        plt.legend()
        plt.savefig(str(C)+'clust_loss'+str(str(main_server.clusters[c].number))+".png")
        # plt.close()   
    "loss mse accuracy"
    for c in range(9):
        tmp1 = []
        tmp2 = []
        for i in range(len(evals_own[c])):
            # print(evals_own[0][i][0])
            tmp1.append(evals_own[c][i][1])
            tmp2.append(evals_diff[c][i][1])
        plt.figure(main_server.clusters[c].number)
        plt.title('MSE : cluster '+str(main_server.clusters[c].number))
        plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
        plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster ' +str(C)+ ' | jacket')
        plt.legend()
        plt.savefig(str(C)+'clust_mse'+str(str(main_server.clusters[c].number))+".png")
        # plt.close()   
    "loss mse accuracy"
    for c in range(9):
        tmp1 = []
        tmp2 = []
        for i in range(len(evals_own[c])):
            # print(evals_own[0][i][0])
            tmp1.append(evals_own[c][i][2])
            tmp2.append(evals_diff[c][i][2])
        plt.figure(main_server.clusters[c].number)
        plt.title('OVERALL ACCURACY : cluster '+str(main_server.clusters[c].number))
        plt.plot(range(len(tmp1)), tmp1, label = 'own_data')
        plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster ' +str(C)+ ' | jacket')
        plt.legend()
        plt.savefig(str(C)+'clust_acc'+str(str(main_server.clusters[c].number))+".png")
        # plt.close()   

    if C == 0:
        acc = []
        loss = []
        mse = []
        for i in range(len(server_own)):
            loss.append(server_own[i][0])
            mse.append(server_own[i][1])
            acc.append(server_own[i][2])
        
        plt.figure()
        plt.title('LOSS of server')
        plt.plot(range(len(loss)), loss, label = 'server loss')
        plt.legend()
        plt.savefig('server_LOSS.png')
        plt.close()   
        
        plt.figure()
        plt.title('MSE of server')
        plt.plot(range(len(mse)), mse, label = 'server mse')
        plt.legend()
        plt.savefig('server_MSE.png')
        plt.close()   
            
        plt.figure()
        plt.title('ACC of server')
        plt.plot(range(len(acc)), acc, label = 'server acc')
        plt.legend()
        plt.savefig('server_ACC.png')            
        plt.close()                     
        
# for c in range(9):
tmp1 = []
tmp2 = []
for i in range(len(server_own)):
    # print(evals_own[0][i][0])
    tmp1.append(1 - server_own[i])
    # tmp2.append(evals_diff[c][i][2])
# plt.figure(main_server.clusters[c].number)
plt.title('OVERALL ACCURACY of a Server')
plt.plot(range(len(tmp1)), tmp1, label = 'server whole data')
# plt.plot(range(len(tmp2)), tmp2, label = 'data of cluster ' +str(C)+ ' | jacket')
plt.legend()
plt.savefig("server_acc_overall.png")