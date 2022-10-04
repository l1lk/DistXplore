import numpy as np
import os

for dataset in ["mnist_lenet4", "fmnist_lenet4", "cifar_resnet", "svhn_resnet"]:
    save_dir = "/data/c/tianmeng/wlt/select_coverage_data/%s"%dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_first_try = True
    for truth in range(10):
        model_select_index = np.load("/data/c/tianmeng/wlt/select_mmd_coverage_index/%s/truth_%s_index.npy"%(dataset,truth), allow_pickle=True)[0]
        # print(model_select_index)
        # crash_select_index = np.load("./select_mmd_coverage_index/cifar_crash_index_v2.npy", allow_pickle=True)
        model_select_len = []
        for index in model_select_index:
            model_select_len.append(len(index))
        iteration_list = np.arange(0,31)
        # print(iteration_list)
        # low_diversity_select_index = []
        # for lenth in model_select_len:
        #     low_diversity_select_index.append(iteration_list[-lenth:])
        # print(low_diversity_select_index)
        # crash_data_dir = "/data1/wlt/class_%s_seed_output_best_mmd_cifar"%truth
        crash_data_dir = "/data/c/tianmeng/wlt/ga_iteration_data/GA_100_logits_%s/class_%s"%(dataset, truth)
        first_try = True
        target_list = np.arange(10)
        target_list = np.delete(target_list, truth)
        for target in target_list:
            target_model_select_index = model_select_index[(target-1)]
            # print("test", target_model_select_index)
            # target_model_select_index = low_diversity_select_index[(target-1)]
        # select_iteration = [2,3,4,5]
            for iteration in target_model_select_index:
                temp_data = np.load(os.path.join(crash_data_dir, "data_%s_%s_%s.npy"%(truth, target, iteration+1)))
                temp_truth = np.load(os.path.join(crash_data_dir, "ground_truth_%s_%s_%s.npy"%(truth, target, iteration+1)))
                if first_try:
                    select_data = temp_data
                    select_truth = temp_truth
                    first_try = False
                else:
                    select_data = np.concatenate((select_data, temp_data), axis=0)
                    select_truth = np.concatenate((select_truth, temp_truth), axis=0)
        shuffle_index = np.random.permutation(len(select_data))
        select_data = select_data[shuffle_index]
        select_truth = select_truth[shuffle_index]
        # print(select_data.shape)
        if all_first_try:
            all_select_data = select_data
            all_select_truth = select_truth
            all_first_try = False
        else:
            all_select_data = np.concatenate((all_select_data, select_data), axis=0)
            all_select_truth = np.concatenate((all_select_truth, select_truth), axis=0)
    print(all_select_data.shape)
    print(all_select_truth.shape)
    np.save(os.path.join(save_dir, "data_high_coverage.npy"), all_select_data)
    np.save(os.path.join(save_dir, "ground_truth_high_coverage.npy"), all_select_truth)
