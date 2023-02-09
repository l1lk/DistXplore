import numpy as np
import os
from collections import Counter

first_try = True
for i in range(1):
    tmp_data = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/temp_data_%s.npy"%i)
    tmp_truth = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/temp_truth_%s.npy"%i)
    if first_try:
        all_data = tmp_data
        all_truth = tmp_truth
        first_try = False
    else:
        all_data = np.concatenate((all_data, tmp_data),axis=0)
        all_truth = np.concatenate((all_truth, tmp_truth), axis=0)
shuffle_index = np.random.permutation(len(all_data))
all_data = all_data[shuffle_index]
all_truth = all_truth[shuffle_index]
all_data = all_data[:1000]
all_truth = all_truth[:1000]
all_data = all_data*255
all_data = all_data.astype(np.uint8)
print(all_data.shape)
print(np.unique(all_data))
print(np.unique(all_truth))
np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/seeds.npy", all_data)
np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/truth.npy", all_truth)