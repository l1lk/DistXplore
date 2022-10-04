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
from collections import Counter
import joblib
import keras
from tensorflow.keras.models import load_model
def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

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

def preprocessing_batch(x_test):
    if np.ndim(x_test)==3:
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test


# def extract_data_features(sub_model, data):
#     # print("shape", data.shape)
#     # sub_model = keras.models.Model(inputs=model.input, 
#     #                                outputs=model.get_layer(index=-2).output)
#     features = sub_model.predict(data)
#     # print("extract feature shape {}".format(features.shape))
#     return features


model = load_model("profile/vgg16_cifar.h5")
sub_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)

save_pth = './best_chromo_output/cifar/'
if not os.path.exists(save_pth):
    os.makedirs(save_pth)
    
for truth_label in range(10):
    target_list = np.arange(10)
    target_list = np.delete(target_list, truth_label)
    # target_list = [truth_label]
    target_mmd_matrix = []
    for target in target_list:

        print('truth: {}, target: {}'.format(truth_label, target))
        # label_train_features = train_feature[np.where(y_train==target)[0]]
        seed = np.load("./single_cluster_seeds/cifar/training_100/class_%s_seed.npy"%truth_label)
        
        # seed = seed.reshape(len(seed), 28,28,1)
        seed = cifar_preprocessing(seed)
        seed_features = sub_model.predict(seed)
        # seed_features = extract_data_features(sub_model, seed)

        mmds_list = []
        # crash_num_list = []
        for iteration in tqdm(range(1,32)):

            temp_data = np.load("./data_ite/ga_best_chromo_iteration/cifar/class_%s/data_%s_%s_%s.npy"%(truth_label, truth_label, target, iteration))
            # crash_num_list.append(len(temp_data))
            
            temp_data = cifar_preprocessing(temp_data)
   
            # temp_features = extract_data_features(sub_model, temp_data)
            temp_features = sub_model.predict(temp_data)

            mmds = mmd.cal_mmd(temp_features, seed_features)
            mmds_list.append(mmds.tolist())
        target_mmd_matrix.append(mmds_list)
    # print(target_mmd_matrix)
    np.save(save_pth + 'best_chromo_truth_%s_mmds.npy'%truth_label, target_mmd_matrix)
