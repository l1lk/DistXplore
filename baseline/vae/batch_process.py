import os
import numpy as np
import glob
import imageio
def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp



# work_path = 'data_all_seed2/cifar/cw_ori/'

# save_pth = 'data_all_seed2/cifar/cw/'
# if not os.path.exists(save_pth):
#     os.makedirs(save_pth)

# for truth_idx in range (10):
#     class_idx = [j for j in range(10) if j != truth_idx]
#     print(class_idx)
#     all_pth = glob.glob(work_path + '*_' + str(truth_idx) + '_target_' + str(class_idx) + '_*.npy') 
    
#     data = None 
#     truth = []
#     for all_p in all_pth:
#         data_ = np.load(work_path + all_p)

#         if data is None:
#             data = data_
#         else:
#             data = np.concatenate((data, data_), axis=0)

#         truth += [int(truth_idx) for i in range(len(data_))]

    
#     truth = np.array(truth)
#     print(data.shape)
#     print(truth.shape)

#     save_name = 'data_' + str(truth_idx) + '_' + str(class_idx) + '.npy'
#     save_truth = 'ground_truth_' + str(truth_idx) + '_' + str(class_idx) + '.npy'

#     # np.save(save_pth + save_name, data)
#     # np.save(save_pth + save_truth, truth)   

'''
vae
'''

work_path = 'vae_svhn_seed2/'
all_pth = os.listdir(work_path)
all_pth.sort()

save_pth = 'data_all/svhn/vae_seed2'
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

# exist_data = np.load('vae_npy/mnist/data.npy')
# exist_label = np.load('vae_npy/mnist/ground_truth.npy')

# print(exist_data.shape)
# print(exist_label.shape)
data = None
truths = []
targets = []
for p in all_pth:
    path = glob.glob(work_path + p + '/*.npy')
    # print(path)

    for npy_pth in path:
        # print(npy_pth)

        split_p = npy_pth.split('/')
        splited = split_p[-1].split('_')

        truth = splited[1]

        # print(truth)
        truths.append(truth)
        # target = split_p[-1][0]

        # # image = imageio.imread(path)
        image = np.load(npy_pth)
        image = np.array(image)
        
        # # data.append(image)
        
        # targets.append(target)
        if data is None:
            data = image
        else:
            data = np.concatenate((data, image), axis=0)

# data = np.concatenate((data, exist_data), axis=0)
truths = np.array(truths)
# truths = np.concatenate((truths, exist_label), axis=0)
print(data.shape)
print(truths.shape)


# np.save(save_pth+'/data.npy', data)
# np.save(save_pth+'/ground_truth.npy', truths)   

