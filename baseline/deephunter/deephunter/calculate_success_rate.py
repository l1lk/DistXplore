import numpy as np
import os

load_dir = "/data/wlt/deephunter/deephunter/deephunter_outputs/svhn_vgg_ga_nbc_iter_5000_efficient/outputs_50/crashes"
file_list = os.listdir(load_dir)
seed_list = []
for file_name in file_list:
    seed_list.append(int(file_name.split("_")[3]))
unique_index = np.unique(seed_list)
print(unique_index)
print(len(np.where(unique_index <=1000)[0]))
