import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from merge_output_def import dissector
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dissector auc")
    parser.add_argument('-mode', help="dataset and model type")
    parser.add_argument('-tech', help="sample generation tech")
    parser.add_argument('-truth', type=int)
    parser.add_argument('-target', type=int)
    args = parser.parse_args()
    # mode = "mnist_lenet4"
    # tech = "bim"
    mode = args.mode
    tech = args.tech
    print(mode, tech)

    crash_save_dir = "/data/c/all_adv_data/svhn_resnet/%s"%tech

    auc_save_dir = "/data/c/dissector/svhn_%s_resnet"%tech
    if not os.path.exists(auc_save_dir):
        os.makedirs(auc_save_dir)
    all_auc_list = []
    for truth in [args.truth]:
        target_list = np.arange(10)
        target_list = np.delete(target_list, truth)
        target_auc_list = []
        for target in [args.target]:
            data = np.load(os.path.join(crash_save_dir, "data_%s_%s.npy"%(truth, target)))
            ground_truth = np.load(os.path.join(crash_save_dir, "ground_truth_%s_%s.npy"%(truth, target)))
            auc_score = dissector(data, ground_truth, "svhn", "resnet20", preprocess=False)
            target_auc_list.append(auc_score)
        all_auc_list.append(target_auc_list)
    all_auc_list = np.array(all_auc_list)
    print(all_auc_list.shape)
    np.save(os.path.join(auc_save_dir, "%s_%s.npy"%(args.truth, args.target)), all_auc_list)
