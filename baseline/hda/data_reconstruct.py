import numpy as np
import os

for i in range(10):
    data = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/class_%s_seed.npy"%i)[:100]
    print(np.unique(data))
    print(data.dtype)
    data = data.astype(np.uint8)
    print(data.dtype)
    print(np.unique(data))
    # data = np.load("/data/c/tianmeng/distribution/single_cluster_seeds/mnist/training_100/class_%s_seed.npy"%i)[:100]
    # print(data.dtype)
    # re_data = []
    # for indv in data:
    #     re_indv = indv * 255
    #     re_data.append(re_indv)
    # re_data = np.array(re_data)
    # print(np.unique(re_data))
    np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/class_%s_seed.npy"%i, data)