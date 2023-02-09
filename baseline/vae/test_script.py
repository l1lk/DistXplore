import numpy as np
import argparse, pickle
import joblib
import mmd_coverage as mmdcov
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
import cal_mmd as mmd
import os
import random
import tqdm
from tensorflow.keras.models import load_model
import AttackSet as att
import scipy.io
import imageio
import glob
from math import nan, isnan
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from keras.utils import np_utils


def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]       
    return temp

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def produce_seed(grouped_x_train, seed_num, model):
    for idx, class_key in enumerate(grouped_x_train):
        
        class_name = class_key
        # -----------svhn-----------
        # if (class_key[1]+class_key[2]) == '10': 
        #     class_name = '0'
        #     trans_label = '9'
        # else:
        #     class_name = class_key[1]
        #     trans_label = int(class_name) - 1
        # -----------svhn-----------

        print('class key: {}'.format(class_key))
        current_data = grouped_x_train[class_key]
        pred = np.argmax(model.predict(mnist_preprocessing(current_data)), axis=1)
        print(pred)
        data_ = [current_data[i] for i in range(len(pred)) if pred[i]==int(class_name)] # trans_label, class_name
        
        tmp_list = random.sample(data_, seed_num)
        
        save_name = "class_" + class_name + "_seed.npy"
        
        seed_pth = 'single_cluster_seeds/fashion_mnist/training_' + str(seed_num) + '_v2'
        if not os.path.exists(seed_pth):
            os.makedirs(seed_pth)
        np.save(os.path.join(seed_pth, save_name), tmp_list)

def preprocess_mmd(grouped_x_train, model):
    preprocessed_grouped_train = []
    for class_key in grouped_x_train:
        train_preprocessed = mnist_preprocessing(grouped_x_train[class_key])
        train_outputs = att.predict(train_preprocessed, model)
        train = train_outputs[-2]
        preprocessed_grouped_train.append(train)
    return preprocessed_grouped_train


def cal_training_mmd(grouped_x_test):
    record_mmds = []
    preprocessed_grouped_train = preprocess_mmd(grouped_x_test)

    for cur_cls in tqdm.tqdm(preprocessed_grouped_train):
        mmds = []
        for other_cls in preprocessed_grouped_train:
            mmds.append(mmd.cal_mmd(cur_cls, other_cls).cpu().detach().numpy())
        record_mmds.append(mmds)


    with open('origin_data_mmd/test_mmd_records.txt', 'w') as f:
                for idx, class_record in enumerate(record_mmds):
                    f.write(str(idx)+'\n')
                    for record in class_record:
                        f.write(str(record)+'\n')
                    f.write('\n')



def cal_acc(grouped_x_train, model):
    pred = []
    for key in grouped_x_train:
        train_ = np.array(grouped_x_train[key])
        outputs = att.predict(cifar_preprocessing(train_), model)   
        tmp_pred = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(key)]
        acc = len(tmp_pred)/len(train_)
        pred.append(acc)
        print('class ' + key + ' accuracy: ' + str(acc))

    print('Average acc is {:.3f}'.format(np.mean(pred)))


model_pth = {
    'vgg16': "profile/vgg16_cifar10.h5",
    'lenet5': "profile/lenet5_mnist.h5",
    'lenet4': 'profile/lenet4_mnist.h5'
}

shape_dic = {
    'vgg16': (32, 32, 3),
    'lenet5': (28, 28, 1)
}


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train[:60000]
    y_train = y_train[:60000]
    x_test = x_test[:10000]
    y_test = y_test[:10000]

   
    grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    return grouped_x_train, grouped_x_test, y_train

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    class_num = len(set(y_train))

    x_train = x_train[:60000]
    y_train = y_train[:60000]
    x_test = x_test[:10000]
    y_test = y_test[:10000]

   
    grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    # gmms, map_train_set =  mmdcov.ini_gmm_training(5, grouped_x_train.reshape(len(grouped_x_train), 784), y_train)
    # map_test_set = mmdcov.map_train_cluster(5, grouped_x_test.reshape(len(grouped_x_test), 784), y_train, gmms)
    return grouped_x_train, grouped_x_test, y_train


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    class_num = len(set(y_train[:,0]))
    
    x_train = x_train[:50000, :, :, :]
    y_train = y_train[:,0][:50000]
    x_test = x_test[:10000, :, :, :]
    y_test = y_test[:,0][:10000]

    grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    return grouped_x_train, grouped_x_test, y_train

def load_svhn():

    x_train = scipy.io.loadmat('dataset/train_32x32.mat')['X'] # 73257
    y_train = scipy.io.loadmat('dataset/train_32x32.mat')['y']

    x_test = scipy.io.loadmat('dataset/test_32x32.mat')['X'] # 26032 
    y_test = scipy.io.loadmat('dataset/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    # print(y_train)
    grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    grouped_x_test = mmdcov.group_into_class(x_test, y_test)
    
    return grouped_x_train, grouped_x_test, y_train


# Robustness: compare defence/retraining effectiveness on same num of clusters but with different mmd distribution.
# Randomly Make Test Group: 
#   - random select n number of clusters from [original crashes]/[ga crashes]/[mixture crashes] to form a test group 
#   - (e.g. for original crashes, form one group with low mmd distribution and one group with high mmd distribution)
#   - Do a Horizontal and Vertical comparsion
# calculate mmd distribution of test group: random select one cluster, using sum(unique mmd)/num of cluster 
#   - no upper bound, only intervals [0,1]/[0,10]
#   - *abundance/each interval
# Manually Make Test Group
#   - random select 

def load_crash(pth):
    crash = None
    seed_pth = os.listdir(pth)
    for p in seed_pth:
        path = os.path.join(pth, p)
        cluster = np.load(path)
        if crash is None:
            crash = cluster
        else:
            crash = np.concatenate((crash, cluster), axis=0)

    return crash

def mmd_group_random(crash, num, size, model):
    
    crashes = []
    for n in range(num):
        crashes.append(random.sample(list(crash), size))
            
    seed_cluster = random.sample(crashes, 1)

    crashes = np.array(crashes)
    seed_cluster = np.array(seed_cluster)
    
    origin_point = att.predict(mnist_preprocessing(seed_cluster[0]), model)[-2]
    
    mmds = []
    for cluster in crashes:
        outputs = att.predict(mnist_preprocessing(cluster), model)[-2]
        mmds.append(mmd.cal_mmd(outputs, origin_point).cpu().detach().numpy())

    
    interval_ = 0.1
    start = 0
    empty = False
    records = []
    density_records = []
    while empty is False:
        tmp_record = 0
        for md in mmds:
            end = start + interval_
            if md > start and md < end:
                tmp_record += 1
                density_records.append(end)
        if tmp_record == 0:
            empty = True
            break
        records.append(tmp_record)
        start+=interval_
    
    
    fig = plt.figure()
    sns.set_style('whitegrid')
    sns.kdeplot(density_records, bw=0.5)

    plt.ylim(0,8)
    plt.xlim(0,9)
    plt.savefig('GA_output_chart/kde_best.png')
    
    print(records)
    log_sum = 0
    for interval in records:
        log_sum += math.log(interval/(num))
    normed = abs(log_sum)/len(records)
    print(log_sum)    
    print(normed)    
    return crashes, 1/normed
  
    




# model = load_model('profile/lenet4_mnist.h5')
model = load_model('profile/lenet5_mnist.h5')
print(model.summary())

# model = load_model('profile/vgg16_cifar10.h5')
# model = load_model('profile/resnet20_cifar10.h5')
# model = load_model('profile/vgg16_svhn.h5')
# model = load_model('profile/resnet20_svhn.h5')
# model = load_model('profile/lenet4_fm.h5')
# model = load_model('profile/lenet5_fm.h5')



'''
save model weights
'''
# model.save_weights('lenet5_fm_weights.h5')
'''
eval seed acc
# '''
# pth = 'single_cluster_seeds/svhn/training_100_v2'
# path = os.listdir(pth).sort()
# for p in path:
#     se = np.load(os.path.join(pth, p))
#     data = np.array(se)
#     output = np.argmax(model.predict(svhn_preprocessing(data)), axis=1)
#     print(output)
#     print(len(output))

# print(model.evaluate(mnist_preprocessing(data), keras.utils.to_categorical(truth, 10)))


'''
model acc
'''
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# # y_train = np_utils.to_categorical(y_train, num_classes=10)
# # y_test = np_utils.to_categorical(y_test, num_classes=10)
# # print(y_test.shape)
# pred = np.argmax(model.predict(mnist_preprocessing(X_test)), axis=1)
# result = [pred[i] for i in range(len(pred)) if pred[i]==y_test[i]]
# print(len(result)/len(y_test))
# # X_test = X_test.reshape(-1, 28, 28, 1)
# # print(model.evaluate(X_test/255., y_test))


# data = np.load('single_cluster_seeds/svhn/training_100/class_1_seed.npy')
# data = np.load('GA_output/GA_100_logits_svhn/100_50/class_0_seed_output_1/crashes/1_class_0_seed_tar1_gt9_pred0_0.271.npy')
# data = np.array(data)

# truth = [1 for i in range(len(data))]

# print(np.argmax(model.predict(svhn_preprocessing(data)), axis=1))
# print(model.evaluate(svhn_preprocessing(data), keras.utils.to_categorical(truth, 10)))


# data = 'GA_output/GA_100_logits_mnist/100_50/class_2_seed_output_4/crashes'
# data_list = os.listdir(data).sort()
# crash = None
# for pth in data_list:
#     # print(pth)
#     crash_ = np.load(os.path.join(data, pth))
    
#     if crash is None:
#         crash = crash_
#     else:
#         crash = np.concatenate((crash, crash_), axis=0)
# print('data_list len {}'.format(len(data_list)))
# print('crash shape test {}'.format(crash.shape))
# outputs = att.predict(mnist_preprocessing(crash), model)   
# tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(2)]

# ori_predict_label = np.argmax(model.predict(mnist_preprocessing(crash)), axis=1)
# target_list = [i for i in ori_predict_label if int(i) == int(4)]
# print(tmp_truth)
# print(target_list)
# print(ori_predict_label)
# print(len(target_list)/len(crash))
# print(len(tmp_truth)/len(outputs[-1]))
'''
show image
'''
# img0 = Image.fromarray(data[3])
# # img0.save(os.path.join(img_output, name + '.png'))
# img0.show()
'''
Compare mmd density between training and crashes
'''
# grouped_x_train, grouped_x_test, y_train = load_mnist()
# ga_crash1 = load_crash('GA_output/GA_100_logits_mnist/100_50/class_1_seed_output_0/crashes')
# ga_crash2 = load_crash('GA_output/GA_100_logits_mnist/100_50/class_2_seed_output_0/crashes')
# bim_crash = load_crash('GA_transfer/mnist/all_adv_data_mnist/bim/data_1_0.npy')
# bim_crash2 = load_crash('GA_transfer/mnist/all_adv_data_mnist/bim/data_2_0.npy')
# cw_crash = load_crash('GA_transfer/mnist/all_adv_data_mnist/cw/data_1_0.npy')
# cw_crash2 = load_crash('GA_transfer/mnist/all_adv_data_mnist/cw/data_2_0.npy')
# pgd_crash = load_crash('GA_transfer/mnist/all_adv_data_mnist/pgd/data_1_0.npy')
# pgd_crash2 = load_crash('GA_transfer/mnist/all_adv_data_mnist/pgd/data_2_0.npy')
# # kmnc_crash1 = load_crash('GA_transfer/mnist/deephunter_crash_mnist/kmnc/data_2_0.npy')
# # kmnc_crash2 = load_crash('GA_transfer/mnist/deephunter_crash_mnist/kmnc/data_2_0.npy')
# # nbc_crash1 = load_crash('GA_transfer/mnist/all_adv_data_mnist/pgd/data_2_0.npy')
# # nbc_crash2 = load_crash('GA_transfer/mnist/all_adv_data_mnist/pgd/data_2_0.npy')


# crashes1, index_ = mmd_group_random(crash1, 100, 30, model) # (crash, clusters numbers, cluster size, model) 
# crashes2, index_ = mmd_group_random(crash2, 100, 30, model)
# training, index_ = mmd_group_random(grouped_x_train['0'], 100, 30, model)

'''
metric test
'''

# crash = load_crash('GA_output/GA_100_logits_mnist/100_50/class_0_seed_output_1/crashes')
# best_crash = load_crash('GA_output/GA_100_logits_mnist_v2/100_50/class_0_seed_output_1/best_mmds')

# # print(crash.shape)
# # print(best_crash.shape)
# crashes, index_ = mmd_group_random(crash, 100, 30, model)
# crashes2, index_2 = mmd_group_random(best_crash, 100, 30, model)
# print(index_)
# print(index_2)

'''
transfer on attack all (bim, cw, pgd) 
'''

# save_pth = 'GA_transfer/mnist/pgd/'
# if not os.path.exists(save_pth):
#     os.makedirs(save_pth)

# accs = []
# portions = []
# data_pth = 'GA_transfer/mnist/all_adv_data_mnist/pgd'
# _pth = os.listdir(data_pth)
# for p in _pth:
#     crash_ = np.load(os.path.join(data_pth, p))
#     split_pth = p.split('_')
#     print(p)
#     outputs = att.predict(crash_, model) 
#     tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(split_pth[-2])]
#     tmp_target = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(split_pth[-1][0])]
#     acc = len(tmp_truth)/len(crash_)
#     portion = len(tmp_target)/len(crash_)
#     # print('acc is {}'.format(acc))
#     # print('target proportion is {}'.format(portion))
#     accs.append(acc)
#     portions.append(portion)

# with open(save_pth + 'acc_records.txt', 'w') as f:
#     for acc in accs:
#         f.write(str(acc)+'\n')
    
# with open(save_pth + 'portion_records.txt', 'w') as f:   
#     for por in portions:
#         f.write(str(por)+'\n')


'''
transfer on GA all
'''
# save_pth =  'GA_transfer/mnist/mmd/'
# if not os.path.exists(save_pth):
#     os.makedirs(save_pth)

# # data_pth = 'GA_transfer/cifar/ga_data'
# data_pth = 'GA_output/GA_100_logits_mnist/100_50'
# crash_list = os.listdir(data_pth)
# accs = []
# target_portion = []
# for pth_ in crash_list: # pth_ = class_0_seed_output_1
#     path = os.path.join(data_pth, pth_)
#     target = pth_[-1]
#     truth = pth_[6]
#     print(pth_)
    
#     crash_pth_list = []
#     crash = None
#     for i in range(1, 32):
#         crash_pth = glob.glob(path + '/crashes/'+str(i)+'_class*.npy')
#         crash_pth_list += crash_pth
        
#         tmp_acc = []
#         tmp_target_portion = []
#         for pth in crash_pth:
#             # print('current iteration {}, current pth is {}'.format(i, pth))
#             crash_ = np.load(pth)
#             if crash is None:
#                 crash = crash_
#             else:
#                 crash = np.concatenate((crash, crash_), axis=0)
   
#     outputs = att.predict(mnist_preprocessing(crash), model)   
#     # print(len(outputs[-1]))
#     tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(truth)]
#     tmp_target = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(target)]
#     accs.append(len(tmp_truth)/len(outputs[-1]))
#     target_portion.append(len(tmp_target)/len(outputs[-1]))
    
# with open(save_pth + 'acc_records.txt', 'w') as f:
#     for acc in accs:
#         f.write(str(acc)+'\n')
    
# with open(save_pth + 'portion_records.txt', 'w') as f:   
#     for por in target_portion:
#         f.write(str(por)+'\n')


'''
transfer on attack (bim, cw, pgd) 
'''
# accs = []
# portions = []
# data_pth = 'GA_transfer/all_adv_data/pgd'
# _pth = os.listdir(data_pth)
# for p in _pth:
#     crash_ = np.load(os.path.join(data_pth, p))
#     split_pth = p.split('_')
#     # print(split_pth)
#     outputs = att.predict(crash_, model) 
#     tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(split_pth[-2])]
#     tmp_target = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(split_pth[-1][0])]
#     acc = len(tmp_truth)/len(crash_)
#     portion = len(tmp_target)/len(crash_)
#     print('acc is {}'.format(acc))
#     print('target proportion is {}'.format(portion))
#     accs.append(acc)
#     portions.append(portion)


# acc_pth = data_pth + '/records/'

# if not os.path.exists(acc_pth):
#     os.makedirs(acc_pth)
# with open(acc_pth + 'records.txt', 'w') as f:
#     f.write(str(np.mean(accs))+'\n')
#     f.write(str(np.mean(portions))+'\n')
#     f.write(str(max(accs))+'\n')
#     f.write(str(min(accs))+'\n')

'''
transfer on GA
'''
# acc_pth =  'GA_transfer/cifar/resnet20_cifar10/acc/'
# portion_pth = 'GA_transfer/cifar/resnet20_cifar10/portion/'
    
# if not os.path.exists(acc_pth):
#     os.makedirs(acc_pth)
# if not os.path.exists(portion_pth):
#     os.makedirs(portion_pth)  

# data_pth = 'GA_transfer/cifar/ga_data'
# crash_list = os.listdir(data_pth)
# for pth_ in crash_list:
#     path = os.path.join(data_pth, pth_)
#     accs = []
#     target_portion = []
#     print(pth_)
#     for i in range(1, 32):
#         crash_pth = glob.glob(path + '/crashes/'+str(i)+'_class*.npy')
        
#         target = pth_[-1]
#         truth = pth_[6]
       
#         tmp_acc = []
#         tmp_target_portion = []
#         for pth in crash_pth:
#             crash_ = np.load(pth)
            
#             outputs = att.predict(cifar_preprocessing(crash_), model)   

#             tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(truth)]
#             tmp_target = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(target)]
#             tmp_acc.append(len(tmp_truth)/len(crash_))
#             tmp_target_portion.append(len(tmp_target)/len(crash_))
#         accs.append(np.mean(tmp_acc))
#         target_portion.append(np.mean(tmp_target_portion))

#     # save 
     
    
#     with open(acc_pth + pth_ + '_acc_records.txt', 'w') as f:
#         for acc in accs:
#             f.write(str(acc)+'\n')
    
#     with open(portion_pth + pth_ + '_portion_records.txt', 'w') as f:   
#         for por in target_portion:
#             f.write(str(por)+'\n')

'''
transfer on deephunter (kmnc, nbc)
'''
# data = np.load('GA_transfer/mnist/deephunter_crash_mnist/nbc/data.npy')
# truth = np.load('GA_transfer/mnist/deephunter_crash_mnist/nbc/ground_trutr.npy')
# data = np.array(data)
# print(model.evaluate(mnist_preprocessing(data), keras.utils.to_categorical(truth, 10)))


# data = np.load('GA_transfer/deephunter_crash/nbc/data.npy')
# label = np.load('GA_transfer/deephunter_crash/nbc/ground_trutr.npy')
# data = np.array(data)
# print(data.shape)
# outputs = att.predict(mnist_preprocessing(data), model)
# print(len(outputs[-1]))

# tmp_truth = []
# for idx, output in enumerate(outputs[-1]):
#     # print(np.argmax(output))
#     # print(int(label[idx]))
#     if np.argmax(outputs[-1]) == int(label[idx]):
#         tmp_truth.append(np.argmax(output))
# # tmp_truth = [np.argmax(i) for i in outputs[-1] if np.argmax(i) == int(label[-2])]    
# # # tmp_truth = [np.argmax(outputs[i][-1]) for i in range(len(outputs)) if np.argmax(outputs[i][-1]) == int(label[i])]
# print(len(tmp_truth)/len(outputs))







'''
Cal AVG on transfer
'''
# records = []
# data_pth = 'GA_transfer/cifar/resnet20_cifar10/portion'
# _pth = os.listdir(data_pth)
# accs = []
# for p in _pth:
    
#     with open(os.path.join(data_pth, p)) as f:
#         cur_line = f.read().splitlines()
#         records += [float(acc) for acc in cur_line if isnan(float(acc)) == False]


# print(np.mean(records))    
# print(max(records))
# print(min(records))
# acc_pth = data_pth + '/records/'

# if not os.path.exists(acc_pth):
#     os.makedirs(acc_pth)
# with open(acc_pth + 'records.txt', 'w') as f:
#     f.write(str(np.mean(records))+'\n')
#     f.write(str(max(records))+'\n')
#     f.write(str(min(records))+'\n')



'''
Cal training acc for each class
'''
# grouped_x_train, grouped_x_test, y_train = load_mnist()
# grouped_x_train, grouped_x_test, y_train = load_cifar10()
# cal_acc(grouped_x_test, model)


'''
Produce multi-cluster seed
'''
# grouped_x_train, grouped_x_test, y_train = load_mnist()
# pth = 'gmms'
# gmm_list = os.listdir(pth)
# gmms = [joblib.load(os.path.join(pth, gmm_li)) for gmm_li in gmm_list]
# # outputs = att.predict(mnist_preprocessing(crash_), model) 
# map_train_set = mmdcov.map_train_cluster_logits(5, grouped_x_train, y_train, gmms)
# seeds = mmdcov.seed_selection(map_train_set, 'multi_cluster_seeds/mnist', 100)
'''
Produce single clutser seed
'''
# grouped_x_train, grouped_x_test, y_train = load_mnist()
# for key_dix in grouped_x_train:
#     images = grouped_x_train[key_dix][:10]
#     for idx, image in enumerate(images):
#         x = Image.fromarray(image)
#         save_pth = 'images/' + key_dix + '/'
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#         x.save(save_pth + str(idx) + '.png')
# print(grouped_x_train.keys())
# produce_seed(grouped_x_train, 100, model)


'''
Group test 
'''
# crash = load_crash('GA_output/GA_100_logits_mnist/100_50/class_0_seed_output_1/crashes')
# print(crash[0].shape)
# print(list(crash)[0].shape)
# group_crash, distribution = mmd_group_random(crash, 300, 100, model)    # (data, group size, cluster size, model)
# print(distribution)





# ------------------------------draft-----------------------------------------------------












# npy1 = np.load("GA_output/GA_100_logits_mnist/100_50/class_0_seed_output_1/crashes/1_class_0_seed_tar1_gt0_pred1_0.589.npy")
# # data = np.load("single_cluster_seeds/mnist/training_100/class_1_seed.npy")
# print(npy1.shape)

# outputs = att.predict(cifar_preprocessing(data), model)   
# output = model.predict(data)
# print(outputs[-1][0])
# print(outputs[-2][0])


# n1 = np.load('GA_output/GA_100_logits_mnist/100_50/class_0_seed_output_1/crashes/1_class_0_seed_tar1_gt0_pred1_0.544.npy')
# n2 = np.load('GA_output/GA_100_logits_mnist/100_50/class_0_seed_output_1/crashes/1_class_0_seed_tar1_gt0_pred1_0.575.npy')
# n3 = np.concatenate((n1, n2), axis=0)
# print(n1.shape)
# print(n2.shape)
# print(n3.shape)


# ground_truth = [np.argmax(i) for i in output]

# print(orig_imgs[0].shape)

# print(np.argmax(output[0]))









# if __name__ == '__main__':



#     parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')            
#     parser.add_argument('-model', help="", choices=['vgg16', 'lenet5'], default='lenet5')

#     args = parser.parse_args()

# set1 = map_train_set[0]['0']
# set2 = map_test_set[0]['0']

# print(mmd.cal_mmd(set1, set2))


# img1 = imageio.imread("GA_output/GA_100_logits_cifar10/100_50/class_0_seed_output_1/imgs/1_class_0_seed_tar1_gt0_pred1_0.544.npy")
# img2 = imageio.imread("GA_output/GA_100_logits_cifar10/100_50/class_0_seed_output_1/imgs/34_class_0_seed_tar1_gt0_pred1_0.993_image_50.png")
# diff = np.linalg.norm(img1 - img2)
# print('--------diff is {} --------'.format(diff))








# print(seed1[0:1].shape)
# # print(seed1[0][0:1])
# f = seed1.flatten()
# print(f[:500])

# seed2 = np.load("single_cluster_seeds/cifar10/training_100/class_0_seed.npy")
# print(seed2.shape)



# seed1 = np.load("dev_seeds_testing/class_0_0_seed.npy")
# # print(seed1[1:2].shape)
# orig_imgs = []
# for idx in range(len(seed1)):
#     orig_imgs.append(seed1[idx])

# # for img in orig_imgs:
# #     print(img.shape)
# diff = np.linalg.norm(orig_imgs[0] - orig_imgs[1])

# ae_sets = np.reshape(seed1, (1, 784))
# print(ae_sets)

# seed1 = np.load("dev_seeds_training/class_0_0_seed.npy")
# seed2 = np.load("dev_seeds_training/class_0_1_seed.npy")
# print(seed2[0:1].shape)
# 
# diff = np.linalg.norm(seed1[0:1] - seed1[1:2])
# print('--------diff is {} --------'.format(diff))

# gmm = joblib.load("gmms/training_data_class_0.gmm")

# seed3 = np.load("mnist_output/prob/mmd_training/queue/id_000049_src_9_4 _seed.npy")
# print(seed3.shape)
# seed2 = np.load("dev_seeds_training/class_0_0_seed.npy")


# print(len(seed2))
# print(seed2.shape)
# print(seed2.shape[0])
# x_test = seed2.reshape(seed2.shape[0], 28, 28, 1)
# x_test2 = x_test.reshape(x_test.shape[0], 28, 28, 1)

# print(x_test)

# x_test = x_test.astype('float32')
# x_test /= 255

# diff = np.linalg.norm(seed2[0:1] - seed2[1:2])
# print('--------diff is {} --------'.format(diff))

# diff = np.linalg.norm(x_test - x_test2)
# print('--------diff is {} --------'.format(diff))



