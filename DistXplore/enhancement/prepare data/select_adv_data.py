import numpy as np
import os

for dataset in ["mnist_lenet4", "fmnist_lenet4", "cifar_resnet", "svhn_resnet"]:
    for adv_tech in ["bim", "pgd", "cw"]:
        first_try = True
        for truth in range(10):
            select_index_matrix = np.load("/data/c/tianmeng/wlt/select_mmd_coverage_index/%s/truth_%s_index.npy"%(dataset, truth), allow_pickle=True)[0]
            index_len_list = []
            for select_index in select_index_matrix:
                index_len_list.append(len(select_index))
            target_list = np.arange(10)
            target_list = np.delete(target_list, truth, axis=0)
            for idx, target in enumerate(target_list):
                adv_data = np.load("/data/c/tianmeng/wlt/all_adv_data/%s/%s/data_%s_%s.npy"%(dataset, adv_tech, truth, target))
                adv_truth = np.load("/data/c/tianmeng/wlt/all_adv_data/%s/%s/ground_truth_%s_%s.npy"%(dataset, adv_tech, truth, target))
                select_number = index_len_list[idx] * 100
                shuffle_index = np.random.permutation(len(adv_data))
                adv_data = adv_data[shuffle_index]
                adv_truth = adv_truth[shuffle_index]
                adv_data = adv_data[:select_number]
                adv_truth = adv_truth[:select_number]
                if first_try:
                    all_data = adv_data
                    all_truth = adv_truth
                    first_try = False
                else:
                    all_data = np.concatenate((all_data, adv_data), axis=0)
                    all_truth = np.concatenate((all_truth, adv_truth), axis=0)
        print(all_data.shape)
        print(all_truth.shape)
        np.save("/data/c/tianmeng/wlt/select_coverage_data/%s/data_%s.npy"%(dataset, adv_tech), all_data)
        np.save("/data/c/tianmeng/wlt/select_coverage_data/%s/ground_truth_%s.npy"%(dataset, adv_tech), all_truth)
