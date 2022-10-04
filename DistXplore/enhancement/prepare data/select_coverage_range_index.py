import numpy as np
import math
import os


def get_high_bound(input_data):
    input_int = int(input_data)
    if (input_data - input_int) > 0.5:
        bound = input_int + 1
    elif (input_data - input_int) < 0.5:
        bound = input_int + 0.5
    return bound

def get_low_bound(input_data):
    input_int = int(input_data)
    if (input_data - input_int) > 0.5:
        bound = input_int + 0.5
    elif (input_data - input_int) < 0.5:
        bound = input_int
    return bound
np.random.seed(0)
for dataset in ["mnist_lenet4", "fmnist_lenet4", "cifar_resnet", "svhn_resnet"]:
    all_mmds_matrix = np.load(os.path.join("/data/c/tianmeng/wlt/ga_iteration_mmds", dataset, "truth_all.npy"))
    all_select_index_matrix = []
    for i in range(10):
        mmds_matrix = all_mmds_matrix[i]
        print(mmds_matrix.shape)
        target_select_index_list = []
        for mmds in mmds_matrix:
            # print(mmds)
            min_mmd = np.min(mmds)
            max_mmd = np.max(mmds)
            # high_bound = get_high_bound(max_mmd)
            # low_bound = get_low_bound(min_mmd)
            high_bound = math.ceil(max_mmd)
            low_bound = int(min_mmd)
            interval_list = np.arange(low_bound, high_bound+0.1, 1)
            # print(low_bound, high_bound, interval_list)
            # print(interval_list)
            select_index_list = []
            for ii in range(len(interval_list)-1):
                in_interval_list = []
                for index, mmd in enumerate(mmds):
                    if mmd > interval_list[ii] and mmd < interval_list[ii+1]:
                        in_interval_list.append(index)
                # print(in_interval_list)
                # print(interval_list[i], interval_list[i+1], in_interval_list)
                if in_interval_list == []:
                    continue
                select_index = np.random.choice(in_interval_list)
                select_index_list.append(select_index)
            print(np.sort(select_index_list))
            
            target_select_index_list.append(np.sort(select_index_list))
        all_select_index_matrix.append(target_select_index_list)
        # np.save("/data1/wlt/select_mmd_coverage_index/svhn_chomo_index_check.npy", all_select_index_matrix)
        save_dir = os.path.join("/data/c/tianmeng/wlt/select_mmd_coverage_index", dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "truth_%s_index.npy"%i), all_select_index_matrix)

# print(np.load("/data/c/tianmeng/wlt/ga_iteration_mmds/mnist_lenet4/truth_0.npy").shape)
# print(np.load("/data/c/tianmeng/wlt/ga_iteration_mmds/mnist_lenet4/mnist_mmds_matrix.npy").shape)


# datasets = ["mnist_lenet4", "fmnist_lenet4", "cifar_resnet", "svhn_resnet"]
# for dataset in datasets:
#     all_data = []
#     for truth in range(10):
#         temp_data = np.load(os.path.join("/data/c/tianmeng/wlt/ga_iteration_mmds", dataset, "truth_%s.npy"%truth))
#         all_data.append(temp_data)
#     all_data = np.array(all_data)
#     print(all_data.shape)
#     np.save(os.path.join("/data/c/tianmeng/wlt/ga_iteration_mmds", dataset, "truth_all.npy"), all_data)