import numpy as np
import keras
import os 
from scipy import io

def load_svhn():

    x_train = io.loadmat('/home/dltest/tianmeng/distribution/dataset/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/home/dltest/tianmeng/distribution/dataset/train_32x32.mat')['y']

    x_test = io.loadmat('/home/dltest/tianmeng/distribution/dataset/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/home/dltest/tianmeng/distribution/dataset/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    print(x_train.shape)
    # print(y_train)
    # grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    # grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    # return grouped_x_train, grouped_x_test, y_train
    return (x_train, y_train), (x_test, y_test)

_, (data, truth) = load_svhn()
truth = truth.reshape(-1)
truth = truth.astype(np.float32)
truth = truth - 1

for idx, tech in enumerate(["hda", "vae"]):
    np.random.seed(idx)
    shuffle_index = np.random.permutation(len(data))
    data = data[shuffle_index]
    truth = truth[shuffle_index]
    data = data[:9000]
    truth = truth[:9000]
    np.save("/data/c/tianmeng/wlt/evaluate_data/svhn/%s/data.npy"%tech, data)
    np.save("/data/c/tianmeng/wlt/evaluate_data/svhn/%s/ground_truth.npy"%tech, truth)