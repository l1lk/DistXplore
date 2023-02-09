import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tensorflow.keras.models import load_model
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from Model4 import Model4
import imageio


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def data_trans_mnist(model, adv_data, adv_label):
    input_height = 28
    input_width = 28
    resize_height = 26
    resize_width = 26
    

    resized_test_data = []
    for data in adv_data:
        resize_data = cv.resize(data, (resize_width, resize_height), interpolation=cv.INTER_NEAREST)
        resized_test_data.append(resize_data)

    resized_test_data = np.array(resized_test_data)

    padding_height = input_height - resize_height
    padding_width = input_width - resize_width

    upper_padding = np.random.randint(0, padding_height)
    left_padding = np.random.randint(0, padding_width)

    pad_test_data = []
    for data in resized_test_data:
        pad_data = cv.copyMakeBorder(data, upper_padding, padding_height - upper_padding, left_padding, padding_width - left_padding, cv.BORDER_CONSTANT, value=0)
        pad_test_data.append(pad_data)
        
    pad_test_data = np.array(pad_test_data)

    pad_test_data = pad_test_data.reshape(-1,28,28,1)
    acc = model.evaluate(pad_test_data, adv_label)[1]
    return acc


# model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/lenet5_softmax.h5")
# adv_data = np.load("../../allconv_mmd_ga_nbc_iter_5000_0/data.npy")
# adv_label = np.load("../../allconv_mmd_ga_nbc_iter_5000_0/ground_truth.npy")
# print(model.evaluate(adv_data/ 255, adv_label))
# print(data_trans_mnist(model, adv_data, adv_label))



# calculate all 
input_pth = 'generated_inputs_Model2/occl/'
work_pth = glob.glob(input_pth + '*.png')
ori_model = load_model("profile/lenet5_mnist.h5")


'''
all
'''
# write on disk 
save_pth = 'defence/data_transformation/mnist/all/'

if not os.path.exists(save_pth):
    os.makedirs(save_pth)

all_mean_list = []
crash = []
truth_ = []


for img_pth in work_pth:
    print('class_output_pth: {}'.format(img_pth))

    pth_name = img_pth.split('\\')[-1]
    pth_name = pth_name.split('_')
    truth = pth_name[-1][0]
    print('truth: {}'.format(truth))

    crash.append(imageio.imread(img_pth))
          
    truth_.append(int(truth))

crash = np.array(crash)  
print(crash.shape)

acc = data_trans_mnist(ori_model, mnist_preprocessing(crash), np.array(truth_))
all_mean_list.append(acc)

with open(save_pth + 'data_trans_records.txt', 'w') as f:
    for item in all_mean_list:
        f.write(str(item)+'\n')

# transfer
outputs = ori_model.predict(mnist_preprocessing(crash))   
tmp_truth = [np.argmax(i) for idx, i in enumerate(outputs) if np.argmax(i) == int(truth_[idx])]
print(len(tmp_truth)/len(outputs))

'''
ite
'''
# write on disk 
# save_pth = 'defence/data_transformation/mnist/ite/'

# if not os.path.exists(save_pth):
#     os.makedirs(save_pth)

# crash_pth = 'GA_output/GA_100_logits_mnist/100_50/class_4_seed_output_6'

# class_mean_list = []

# for ite in range(1, 32):
#     crashes_pth = glob.glob(crash_pth + '/crashes/' + str(ite) + '_class*.npy')
#     ite_class_mean_list = []
        
#     for ith_pth in crashes_pth:
#         pth_name = ith_pth.split('\\')[-1]
#         print(pth_name)
#         pth_name = pth_name.split('_')
#         # print(pth_name)
#         # ite_idx = pth_name[0]
#         # target = pth_name[4][3]
#         truth = pth_name[2]
            
#         all_data = np.load(ith_pth)

#         truth_ = [int(truth) for i in range(all_data.shape[0])]
  
#         acc = data_trans_mnist(ori_model, all_data, np.array(truth_))
#         # print('out is {}'.format(out))
#         ite_class_mean_list.append(acc)

#     class_mean_list.append(np.mean(ite_class_mean_list))
    
   

# with open(save_pth + crash_pth.split('/')[-1] +'data_trans_ecords.txt', 'w') as f:
#     for mean_list in class_mean_list:
#         f.write(str(mean_list)+'\n')
  