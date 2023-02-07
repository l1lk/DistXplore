import numpy as np

data_dir = "mnist_lenet4_ga_kmnc_iter_5000_0"
for data_dir in ["mnist_lenet4_ga_kmnc_iter_5000_0", "mnist_lenet4_ga_nbc_iter_5000_0", "fmnist_lenet4_ga_kmnc_iter_5000_0", "fmnist_lenet4_ga_nbc_iter_5000_0", "cifar_resnet_ga_kmnc_iter_5000_0","cifar_resnet_ga_nbc_iter_5000_0","svhn_resnet_ga_kmnc_iter_5000_0","svhn_resnet_ga_nbc_iter_5000_0"]:
    data = np.load("./cdataset_output/%s/data.npy"%data_dir)
    truth = np.load("./cdataset_output/%s/ground_truth.npy"%data_dir)
    shuffle_index = np.random.permutation(len(data))
    data = data[shuffle_index]
    truth = truth[shuffle_index]
    data = data[:1000]
    truth = truth[:1000]
    np.save("./cdataset_output/%s/data.npy"%data_dir, data)
    np.save("./cdataset_output/%s/ground_truth.npy"%data_dir, truth)
