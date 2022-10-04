import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from sklearn.mixture import GaussianMixture
from keras.datasets import mnist
import collections
from tqdm import tqdm
import random

import glob
import cal_mmd as mmd
import argparse, pickle
import foolbox
import numpy as np
import torchvision.models as models
from collections import Counter
import joblib
import keras
import get_model

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


def preprocessing_batch(x_test):
    if np.ndim(x_test)==3:
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test


def extract_data_features(model, data, preprocess=True):
    print("test", data.shape)
    if preprocess:
        data = preprocessing_batch(data)
    print("test:", data.shape)
    sub_model = keras.models.Model(inputs=model.input, 
                                   outputs=model.get_layer(index=-2).output)
    features = sub_model.predict(data)
    print("extract feature shape {}".format(features.shape))
    return features


model = get_model.cifar_resnet20()
save_dir = "/data/c/tianmeng/wlt/ga_iteration_mmds/cifar_resnet"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for truth_label in [0,1,2,3,4,5,6,7,8,9]:
    target_list = np.arange(10)
    target_list = np.delete(target_list, truth_label)
    target_mmd_matrix = []
    for target in target_list:
        seed = np.load("/home/dltest/tianmeng/distribution/single_cluster_seeds/cifar/training_100/class_%s_seed.npy"%target)
        # seed = seed.reshape(len(seed), 28,28,1)
        # seed = seed / 255.
        seed = cifar_preprocessing(seed)
        seed_features = extract_data_features(model, seed, preprocess=False)
        mmds_list = []
        crash_num_list = []
        for iteration in range(1,32):
            # temp_data = np.load("./class_9_seed_output_best_mmd_cifar/data_9_%s_%s.npy"%(target, iteration))
            # temp_data = np.load("./ga_cifar_10_best_mmd/class_%s_seed_output_best_mmd_cifar_10/data_%s_%s_%s.npy"%(truth_label, truth_label, target, iteration))
            temp_data = np.load("/data/c/tianmeng/wlt/ga_iteration_data/GA_100_logits_cifar_resnet/class_%s/data_%s_%s_%s.npy"%(truth_label, truth_label, target, iteration))
            crash_num_list.append(len(temp_data))
            # temp_data = np.load("D:/ga_cifar_best/class_0_seed_output_1_check/best_mmds/6_mmd4.303.npy")
            temp_data = cifar_preprocessing(temp_data)
            # temp_data = temp_data.reshape(len(seed), 28, 28, 1)
            # temp_data = temp_data / 255.
            temp_features = extract_data_features(model, temp_data, preprocess=False)
            mmds = mmd.cal_mmd(temp_features, seed_features)
            mmds_list.append(mmds.tolist())
        target_mmd_matrix.append(mmds_list)
    # print(target_mmd_matrix)
    # np.save("./ga_cifar_10_best_mmd_mmds/cifar_iteration_best_crash_%s_mmds.npy"%truth_label, target_mmd_matrix)
    np.save(os.path.join(save_dir, "truth_%s.npy"%truth_label),target_mmd_matrix)