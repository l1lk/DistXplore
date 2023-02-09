import numpy as np
import os
import cv2

# save_dir = "./cifar_output_check_4/HDA_mse"

# data_list = []
# truth_list = []
# for index in range(10):
    
#     img_list = os.listdir(os.path.join(save_dir, "AEs_%s"%index))
#     for img_file in img_list:
#         if img_file.split(".")[-1] == "npy":
#             # data = cv2.imread(os.path.join(save_dir, "AEs_%s"%index, img_file))
#             data = np.load(os.path.join(save_dir, "AEs_%s"%index, img_file))
#             truth = img_file.split("_")[1]
#             data_list.append(data)
#             truth_list.append(truth)
# data_list = np.array(data_list)
# truth_list = np.array(truth_list)
# print(data_list.shape)
# print(truth_list.shape)
# print(np.unique(data_list))
# np.save("./new_hda_cifar_check_all_data.npy", data_list)
# np.save("./new_hda_cifar_check_all_gtruth.npy",truth_list)

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

a = np.ones((1,32,32,3),dtype=np.float64)*0.
print(svhn_preprocessing(a))