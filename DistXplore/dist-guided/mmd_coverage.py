import numpy as np
from sklearn.mixture import GaussianMixture
from keras.datasets import mnist
import collections
from tqdm import tqdm
import random
import os
import glob
import cal_mmd as mmd
import argparse, pickle
import foolbox
import numpy as np
import torchvision.models as models
import joblib

def preprocessing_batch(x_test):
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def preprocessing_batch_attack(x_test):
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def createBatch(x_batch, batch_size, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batch_num = len(x_batch) / batch_size
    batches = np.split(x_batch, batch_num, axis=0)
    for i, batch in enumerate(batches):
        test = np.append(batch, batch, axis=0)
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), test)

def group_into_class(data, labels):
    print("Grouping samples into class")
    dic = {}
    for i, label in tqdm(enumerate(labels[:len(data)])):
        key = str(label)
        if key in dic:
            dic[key].append(data[i])
        else:
            dic[key] = []
            dic[key].append(data[i])
    dic = collections.OrderedDict(sorted(dic.items()))
    print("Grouping complete")
    return dic

# return list[{"cluster_1": samples, ... , "cluster_n_components": samples}] index: ith class
def map_train_cluster(n_components, grouped_test_data, train_label, gmms):
    class_sets = []
    seed_class_sets = {}
    
    print("Start mapping")

    # for each class 
    for i in tqdm(range(len(set(train_label)))):
        class_key = str(i)
        
        # for each cluster in each class
        for k in range(n_components):
            key = str(k)
            tmp_test = grouped_test_data[class_key]
            result = gmms[i].predict(tmp_test)
            
            for j, re in enumerate(result):
                tmp_iamge = tmp_test[j] 
                tmp_iamge = tmp_iamge.reshape(28, 28)
                if (re == k):
                    if key in seed_class_sets:
                        seed_class_sets[key].append(tmp_iamge)
                    else:
                        seed_class_sets[key] = []
                        seed_class_sets[key].append(tmp_iamge)
        
        # print(seed_class_sets)
        class_sets.append(seed_class_sets.copy())
       
        seed_class_sets.clear()
        
    print("Mapping complete")
    return class_sets

def map_train_cluster_logits(n_components, grouped_test_data, train_label, gmms):
    class_sets = []
    seed_class_sets = {}
    
    print("Start mapping")

    # for each class 
    for i in tqdm(range(len(set(train_label)))):
        class_key = str(i)
        
        # for each cluster in each class
        for k in range(n_components):
            key = str(k)
            tmp_test = grouped_test_data[class_key]
            tmp_test = np.array(tmp_test)
            result = gmms[i].predict(tmp_test.reshape(tmp_test.shape[0], 784))
            
            for j, re in enumerate(result):
                tmp_iamge = tmp_test[j] 
                tmp_iamge = tmp_iamge.reshape(28, 28)
                if (re == k):
                    if key in seed_class_sets:
                        seed_class_sets[key].append(tmp_iamge)
                    else:
                        seed_class_sets[key] = []
                        seed_class_sets[key].append(tmp_iamge)
        
        # print(seed_class_sets)
        class_sets.append(seed_class_sets.copy())
       
        seed_class_sets.clear()
        
    print("Mapping complete")
    return class_sets




def ini_gmm_training(n_components, grouped_train_data, train_label):
    gmms = []
    class_sets = []
    seed_class_sets = {}
    
    print("Start initializing gmms and training datat clusters")

    for i in range(len(set(train_label))):
        class_key = str(i)
        gmm = GaussianMixture(n_components=n_components, max_iter=2000, random_state=0)
        
        for k in tqdm(range(n_components)):
            key = str(k)
            tmp_test = grouped_train_data[class_key]
            result = gmm.fit_predict(tmp_test)
            
            for j, re in enumerate(result):
                tmp_iamge = tmp_test[j] 
                tmp_iamge = tmp_iamge.reshape(28, 28)
                if (re == k):
                    if key in seed_class_sets:
                        seed_class_sets[key].append(tmp_iamge)
                    else:
                        seed_class_sets[key] = []
                        seed_class_sets[key].append(tmp_iamge)
        
        class_sets.append(seed_class_sets.copy())
        seed_class_sets.clear()

        gmms.append(gmm)

    print ("Initializing complete")
    return gmms, class_sets

# select n seeds from testing data
def seed_selection(map_test_set, output_path, num):
    seeds_class = []
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # for each class_clusters
    for i, seed_class_sets in enumerate(map_test_set):
        seed_list = []
        
        # for each clsuters 
        for cluster_index in seed_class_sets:
            # print(len(seed_class_sets[cluster_index])) # --debug-- check how many elements in each clusters
            if num > len(seed_class_sets[cluster_index]):
                tmp_list = seed_class_sets[cluster_index]
                for idx in seed_class_sets:
                    if len(tmp_list) < num:
                        tmp_list += seed_class_sets[idx]
                if len(tmp_list) > num:
                    range_ = len(tmp_list) - num
                    del tmp_list[:-range_]
                seed_list.append(tmp_list)
            else:
                tmp_list = random.sample(seed_class_sets[cluster_index], num)
                seed_list.append(tmp_list)

            save_name = "class_" + str(i) + "_" + str(cluster_index) + "_seed.npy"
            np.save(os.path.join(output_path, save_name), tmp_list)
        seeds_class.append(seed_list)
    
    print("Seed selection complete")
    return seeds_class
    

# for foolbox
def adv_attack(seeds, labels, method):
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    distance = foolbox.distances.Linfinity

    if method == "fgsm":
        attack = foolbox.attacks.FGSM(fmodel, distance=distance, )
        adversarials = attack(seeds, labels, unpack=True)
        return adversarials
    elif method == "pgd":
        attack = foolbox.attacks.PGD(fmodel, distance=distance)
        adversarials = attack(seeds, labels, unpack=True)
        return adversarials
    elif method == "cw":
        attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        adversarials = attack(seeds, labels, unpack=True)
        return adversarials

# for deephunter 
def load_aes(ae_componets, ppath, y_train, gmms):
    print("Start predicting AEs clusters")

    ground_truth_path = glob.glob(ppath + "crash_labels/id*.npy")
    ground_truth_table = []
    for path in ground_truth_path:
        ground_truth_table.append(np.load(path))

    ae_sets_path = glob.glob(ppath + "crashes/id*.npy")
    ae_sets = []
    for path in ae_sets_path:
        ae_sets.append(np.load(path))
    
    ae_sets = np.reshape(ae_sets, (len(ae_sets), 784))
    grouped_aes = group_into_class(ae_sets, ground_truth_table)
    print(grouped_aes.keys())
    map_ae_set = map_train_cluster(ae_componets, grouped_aes, y_train, gmms)
    
    print("Cluster complete")
    return map_ae_set 


def mmd_in_interval(ae, interval):
    for i in range(len(interval)-1):
        if ae >= interval[i] and ae < interval[i+1]:
            return i 
        else:
            if i == len(interval)-2: 
                return 99
            else:
                continue
        
            
def cal_coverage(ae_componets, map_ae_set, map_train_set):
    print("Start calculating coverage")
    
    coverage = []

    # map_ae_set: [seed_sets for class 1, ..., seed_sets for class 9]
    # seed_class_sets: {"cluster_1": samples, ..., "cluster_n_component": samples}
    for i, seed_class_sets in enumerate(map_ae_set):
            class_index = i

            # counter for recording number of mmds in grid
            mmd_in_grid = 0

            # store mmds for each ae
            class_mmds = []   
            
            # store all ne that are in class i 
            nes = []
            ne_class_sets = map_train_set[class_index]
            for cluster_index in ne_class_sets: 
                nes.append(ne_class_sets[cluster_index])

            # for each ae clsuter
            for cluster_index in seed_class_sets:
                ae = seed_class_sets[cluster_index]
                mmds = []
                # print("ae: {}".format(len(ae))) # --debug--
                
                # for each ae cluster, calculate mmd to [c1, c2, c3, c4, c5]  
                for ne in nes:     
                    mmds.append(mmd.cal_mmd(ae, ne))
                   
                mmds = np.array(mmds)
                # print("mmds: {}".format(mmds)) # --debug--
                class_mmds.append(mmds)
            
            class_mmds = np.array(class_mmds)
            print("class mmds: {}".format(class_mmds))
            
            # after calculating all mmds of aes in class i, find max_mmd and calculate intervals
            grid = []
            column_interval_counter = []
            # global 
            for i in range(ae_componets):
                interval_max = 0
                interval_max = max(class_mmds[:,i])
                interval_min = min(class_mmds[:,i])
                intervals = interval_max/10
                grid.append(np.arange(0, interval_max+intervals+0.1, intervals).tolist())
                column_interval_counter.append([])
            
            # print("grid: {}".format(grid)) # --debug--
            
            # calculate the coverage for each class
            for mmds in class_mmds:
                for index in range(len(mmds)):
                    column = mmds[index]                 # get ae's mmd to [index: c1, c2, c3, c4, c5]
                    column_interval = grid[index]        # get grid [index: c1_intervals, c2_intervals, ... ]

                    tmp_mmd_index = mmd_in_interval(column, column_interval)

                    if tmp_mmd_index < 99: # in intervals 
                        if tmp_mmd_index in column_interval_counter[index]: # continue if repeated
                            continue
                        else:
                            column_interval_counter[index].append(tmp_mmd_index) # add to counter if no repeat
                            mmd_in_grid += 1
            
            print("mmd_in_grid: {}".format(mmd_in_grid))
            mmd_in_grid = mmd_in_grid/pow(10, ae_componets)
            coverage.append(mmd_in_grid)
            print("Coverage for class {} is {}".format(class_index, mmd_in_grid))
    return coverage





if __name__ == '__main__':
    '''
    config
    '''
    parser = argparse.ArgumentParser(description='Coverage calculation for perturbation methods')

    parser.add_argument('-d', help="dataset", choices=['mnist', 'cifar'], default='mnist')

    parser.add_argument('-ncl', help="number of training data clusters", type=int, default=5)
    parser.add_argument('-acl', help="number of mutated/adv data clusters", type=int, default=5)

    parser.add_argument('-op', help="function opition", choices=['0', '1'], default='0') # 0: generate seeds for perturbation 1: calculate coverage
    parser.add_argument('-m', help="perturbation method", choices=['deephunter', 'pgd', 'fgsm', 'cw'], default='deephunter')
    parser.add_argument('-snum', help="seed number", type=int, default=1)

    parser.add_argument('-spath', help="output seed path") # dev_seeds/ --debug--
    parser.add_argument('-ppath', help="input perturbed path") # ../Deephunter/deephunter/mnist_output/prob/kmnc/queue/ --debug--
    
    parser.add_argument('-gmmpath', help="load fitted gmm path")
    
    args = parser.parse_args()


    
    '''
    1. clustering training data <- GMM  e.g. MNIST, 4 clusters 
    2. Testing data <- GMM(training data)
    
    '''
    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train[:60000]
        y_train = y_train[:60000]
        x_test = x_test[:10000]
        y_test = y_test[:10000]

        x_train = np.reshape(x_train, (len(x_train), 784))   # reshape for GMM 
        grouped_x_train = group_into_class(x_train, y_train)

        x_test = np.reshape(x_test, (len(x_test), 784))
        grouped_x_test = group_into_class(x_test, y_test)
    
        # cluster training set
        
        class_num = len(set(y_train))
        
        # train or load gmms 
        if args.gmmpath is None:
            gmms, map_train_set =  ini_gmm_training(args.ncl, grouped_x_train, y_train)
            for idx, gmm in enumerate(gmms):
                joblib.dump(gmm, "gmms/training_class_"+str(idx)+".gmm")
            np.save(map_train_set, "gmms/map_train_set.npy")
        else:
            gmm_list = os.listdir(args.gmmpath)
            gmms = [joblib.load(gmm_li) for gmm_li in gmm_list]
            map_train_set = map_train_cluster(args.ncl, grouped_x_train, y_train, gmms)
        # para: n_componets, test, train, label
        map_test_set = map_train_cluster(args.ncl, grouped_x_test, y_train, gmms)

    
    '''
    op == 0:    3. seeds selection <- select 5 random seeds from testing data
                save to spath
    
    op == 1:
                4. AEs <- Perturbation method: DeepHunter/FGSM/CW/SGD
                load from ppath

                5. Cluster AE sets <- 1) GMM.fit_predict(AE) 2) GMM(training data), 5/10 clusters 

                6. Calculate coverage of each AE sets in each class <- grid[ori, interMMD…, maxMMD] 
    '''
    if args.op == '0':
        if args.m == 'deephunter':
            seeds = seed_selection(map_test_set, args.spath, args.snum) # map_train_set
        else: # "pgd", "fgsm", "cw"
            seeds = seed_selection(map_test_set, args.spath, args.snum)
            perturbed_data = []
            perturbed_label = []
            for cls in range(class_num):
                seed_in_class = seeds[cls]
                for cluster in seed_in_class:
                    labels = np.full(len(cluster), cls)     # create labels for each seed
                    labels = np.array(labels)

                    cluster = np.array(cluster)
                    cluster = preprocessing_batch_attack(cluster)

                    advs = adv_attack(cluster, labels, args.m)
                    # print("adv shape {}".format(advs.shape))
                    perturbed_data.append(advs)
                    perturbed_label.append(labels)
            
           
            perturbed_label = list(np.concatenate(perturbed_label).flat)  

            perturbed_data = np.array(perturbed_data)
            perturbed_data = np.reshape(perturbed_data, (perturbed_data.shape[0], 784)) 
            print(perturbed_data.shape)
            grouped_perturbed = group_into_class(perturbed_data, perturbed_label)
            # print(grouped_perturbed)
            map_ae_set = map_train_cluster(args.acl, grouped_perturbed, y_train, gmms)
            coverage = cal_coverage(args.acl, map_ae_set, map_train_set)
            print("Average coverage is {}".format(sum(coverage)/class_num))

    elif args.op == '1':
        map_ae_set = load_aes(args.acl, args.ppath, y_train, gmms)
        coverage = cal_coverage(args.acl, map_ae_set, map_train_set)
        avg_coverage = sum(coverage)/class_num
        print("Average coverage is {}".format(avg_coverage))

   

    
    # --- Coverage calculation process---
    # for each class
    # for each AE set, mmds.append(ae to training cluster)
    # store each mmd into [c1, c2, c3, c4, c5]  (c1: cluster_1)
    # for [c1-c5] sort(mmds) find c1-c5_max_mmd
    # 10 divisions, each interval = c1-c5_max_mmd - (c1-c5_min_mmd or 0)/10
    # Falls intermediate_mmd into invervals -> for c1 (for c2 (for c3 (for c4 (for c5)))) if in [0 - 9] intervals, mmd_in_grid += 1
    # calculate coverage by mmd_in_grid/10^5



    '''
    --------------------------test block--------------------------
    '''
    # some debug command
    # diff = np.linalg.norm(map_train_set[0]['0'][0] - map_train_set[0]['0'][2])
    # print('--------diff is {} --------'.format(diff))


    # # ------ take x samples for the test ------
    # test = x_train[:200]
    # label = y_train[:200]
    # test = np.reshape(test, (len(test), 784))
    # # print(test[0])
    # # print(test[0].shape)
    # # group each sample into its belonged class 
    # test_1 = group_into_class(test, label)
    # # print(test_1) --debug--

    # # print(test_1['3'][0].reshape(28, 28))

    # # cluster training data into 5 clusters for each class
    # gmm_3 = GaussianMixture(n_components=n_componets, tol=1e-9, max_iter=2000, random_state=0).fit(test_1['3'])
    # print(gmm_3.predict(test_1['3']))
    # # print(gmm.means_[0]) --debug--

    
    # test2 = x_test[:200]
    # label2 = y_test[:200]
    # test2 = np.reshape(test2, (len(test2), 784))
    # test_2 = group_into_class(test2, label2)
    # print(gmm_3.predict(test_2['0']))

    # class_sets = []
    # seed_class_sets = {}
   
    # for i in range(5):
    #     result = gmm_3.predict(test_2['0'])
    #     key = str(i)
    #     for j, re in enumerate(result):
    #         if (re == i):
    #             if key in seed_class_sets:
    #                 seed_class_sets[key].append(test_2['1'][j])
    #             else:
    #                 seed_class_sets[key] = []
    #                 seed_class_sets[key].append(test_2['1'][j])

    # # print(seed_class_sets["1"])
    #     # data = test_2['0'][result]
    #     # print(data)
    
    # # n_componets, test, train, label
    # test_set = map_test_data(n_componets, test_2, test_1, label)
    # print(len(test_set))
    # print(test_set[0]["0"][0].shape)
    '''
    --------------------------test block--------------------------
    '''           