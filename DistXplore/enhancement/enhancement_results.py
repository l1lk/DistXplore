import numpy as np
import os 

for mode in ["mnist", "fmnist", "cifar", "svhn"]:
    ga_results = np.load("/data/c/tianmeng/wlt/enhancement_results/%s_ga_acc.npy"%mode)
    else_results = np.load("/data/c/tianmeng/wlt/enhancement_results/%s_acc.npy"%mode)

    temp_matrix = np.zeros((6,6))
    for i in range(6):
        temp_matrix[i, 0] = ga_results[i]
    print(temp_matrix)

    for i in range(6):
        for j in range(5):
            temp_matrix[i][j+1] = else_results[i][j]
    print(temp_matrix)
    np.save("./%s_acc_matrix.npy"%mode, temp_matrix)

print(np.load("/data/c/tianmeng/wlt/enhancement_results/cifar_acc.npy"))