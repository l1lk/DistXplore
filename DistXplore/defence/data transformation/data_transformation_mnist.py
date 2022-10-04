import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


def data_trans_mnist(model, adv_data, adv_label):
    input_height = 28
    input_width = 28
    resize_height = 26
    resize_width = 26
    # adv_data = cifar_preprocessing(adv_data)

    resized_test_data = []
    for data in adv_data:
        resize_data = cv.resize(data, (resize_width, resize_height), interpolation=cv.INTER_NEAREST)
        resized_test_data.append(resize_data)

    resized_test_data = np.array(resized_test_data)

    padding_height = input_height - resize_height
    padding_width = input_width - resize_width

    upper_padding = np.random.randint(0, padding_height)
    left_padding = np.random.randint(0, padding_width)

    pad_test_data = []
    for data in resized_test_data:
        pad_data = cv.copyMakeBorder(data, upper_padding, padding_height - upper_padding, left_padding, padding_width - left_padding, cv.BORDER_CONSTANT, value=0)
        pad_test_data.append(pad_data)
        
    pad_test_data = np.array(pad_test_data)

    pad_test_data = pad_test_data.reshape(-1,28,28,1)
    # print(np.unique(pad_test_data))
    acc = model.evaluate(pad_test_data, adv_label)[1]
    return acc


# model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/cifar10_vgg_model.194.h5")
model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/lenet5_softmax.h5")
# model = keras.models.load_model("")
truth_acc = []
for truth in range(10):
    target_list = np.arange(10)
    target_list = np.delete(target_list, truth)
    target_acc = []
    for target in target_list:
        adv_data = np.load("D:/distribution-aware-data/defesnse crashes/mnist/cw/seed_v1/data_%s_%s.npy"%(truth, target))
        adv_label = np.load("D:/distribution-aware-data/defesnse crashes/mnist/cw/seed_v1/ground_truth_%s_%s.npy"%(truth, target))
        # adv_data = np.load("D:/distribution-aware-data/defesnse crashes/mnist/nbc/data.npy")
        # adv_label = np.load("D:/distribution-aware-data/defesnse crashes/mnist/nbc/ground_truth.npy")
        # adv_data = adv_data / 255.
        # shuffle_index = np.random.permutation(len(adv_data))
        # adv_data = adv_data[shuffle_index]
        # adv_label = adv_label[shuffle_index]
        # adv_data = adv_data[:1000]
        # adv_label = adv_label[:1000]
        # adv_data = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/data.npy")
        # adv_label = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/ground_truth.npy")
        print(adv_data.shape)
        # print(cifar_preprocessing(adv_data).shape)
        # print(model.evaluate(cifar_preprocessing(adv_data), keras.utils.to_categorical(adv_label,10)))
        acc = data_trans_mnist(model, adv_data, adv_label)
        print(truth, target, acc)
        target_acc.append(acc)

    truth_acc.append(target_acc)
    
# np.save("./fmnist_kmnc_trans.npy", truth_acc)
print(np.average(truth_acc))
# df = pd.DataFrame(truth_acc)
# df.to_excel("./cw_cifar_acc.xlsx", sheet_name="0")




