import numpy as np
import os

load_dir = "/home/dltest/tianmeng/wlt/HDA-Testing-main/cifar10_output/hdaselece_HDA_mse/AEs_cifarvgg_0"
data_list = os.listdir(load_dir)
all_data = []
all_truth = []
for file_name in data_list:
    tmp_data = np.load(os.path.join(load_dir, file_name))
    tmp_truth = int(file_name.split("_")[1])
    all_data.append(tmp_data)
    all_truth.append(tmp_truth)
all_data = np.array(all_data)
all_truth = np.array(all_truth)
all_data = np.transpose(all_data, (0,2,3,1))
print(all_data.shape)
print(np.unique(all_data))
all_data = all_data* 255
all_data = all_data.astype(np.int32)
print(np.unique(all_data))
print(all_truth.shape)    
print(np.unique(all_truth))
np.save("hda_cifarvgg_data.npy", all_data)
np.save("hda_cifarvgg_truth.npy", all_truth)