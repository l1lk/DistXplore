from random import shuffle
import keras
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


score_list = []
# for model_prefix in ["high_coverage", "bim", "pgd", "cw","kmnc", "nbc"]:
for model_prefix in ["hda", "vae"]:
    model = keras.models.load_model("/data/c/retrain_models/fmnist_lenet4/%s_ft_model_20_all.h5"%model_prefix)
    temp_score_list = []
    for truth in range(10):
        target_list = np.arange(10)
        target_list = np.delete(target_list, truth)
        for target in target_list:
            temp_data = np.load("/data/c/evaluate_data/fmnist_lenet4_v2/data_%s_%s.npy"%(truth, target))
            temp_truth = np.load("/data/c/evaluate_data/fmnist_lenet4_v2/ground_truth_%s_%s.npy"%(truth, target))
            temp_data = temp_data /255.
            shuffle_index = np.random.permutation(len(temp_data))
            temp_data =  temp_data[shuffle_index]
            temp_truth = temp_truth[shuffle_index]
            temp_data = temp_data[:2000]
            temp_truth = temp_truth[:2000]
            # temp_data = cifar_preprocessing(temp_data)
            # temp_truth = keras.utils.to_categorical(temp_truth, 10)
            score = model.evaluate(temp_data, temp_truth)[1]
            temp_score_list.append(score)
    score_list.append(np.average(temp_score_list))
print(score_list)
