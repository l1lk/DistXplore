import numpy as np
import os

first_try = True
for i in range(25):
    data_dir = "./cdataset_output/svhn_resnet_ga_kmnc_iter_5000_0_%i"%i
    temp_data = np.load(os.path.join(data_dir, "data.npy"))
    temp_truth = np.load(os.path.join(data_dir, "ground_truth.npy"))
    if first_try:
        all_data = temp_data
        all_truth = temp_truth
        first_try = False
    else:
        all_data = np.concatenate((all_data, temp_data), axis=0)
        all_truth = np.concatenate((all_truth, temp_truth), axis=0)
print(all_data.shape)
print(all_truth.shape)
np.save(os.path.join("./cdataset_output", "svhn_kmnc_data.npy"), all_data)
np.save(os.path.join("./cdataset_output", "svhn_kmnc_truth.npy"), all_truth)
