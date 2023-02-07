import keras
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import shutil
import copy
# import matplotlib.pyplot as plt
from collections import Counter

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD
from scipy import io
import get_model
from get_model import svhn_vgg16


def get_svhn_model():
    model = Sequential()
    weight_decay = 0.0005
    model.add(InputLayer(input_shape=(32,32,3)))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp   

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def crash_dataset(c_out_dir):
    c_labels = []
    c_imgs = []
    # c_out_dir = os.path.join(crash_outdir)
    label_dir = os.path.join(c_out_dir, "crash_labels")
    img_dir = os.path.join(c_out_dir, "crashes")
    label_list = os.listdir(label_dir)
    img_list = os.listdir(img_dir)
    for label in label_list:
        tmp_label = np.load(os.path.join(label_dir, label))
        c_labels.append(tmp_label) 
    for img in img_list:
        tmp_img = np.load(os.path.join(img_dir, img))
        # print("1", np.unique(tmp_img))
        c_imgs.append(tmp_img)
    return np.array(c_imgs), np.array(c_labels)

def filter_dataset(crash_outdir, model_dir, dataset_outdir, dir_list, original_seed):
    # model = keras.models.load_model(model_dir)
    # model = get_svhn_model()
    # model.load_weights("./utils/svhn_vgg16_weight.h5")
    # sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model = get_model.svhn_vgg16()
    # model = svhn_vgg16()
    # model = get_model.fmnist_lenet4()
    # print("Model Summary: ")
    # print(model.summary())
    first_try = True
    for crash_dir in dir_list:
        c_out_dir = os.path.join(crash_output_dir, crash_dir)
        temp_cimgs, temp_clabels = crash_dataset(c_out_dir)
        print(temp_cimgs.shape)
        print(temp_clabels.shape)
        # print("test",np.unique(temp_cimgs))
        if first_try:
            cimgs = temp_cimgs
            clabels = temp_clabels
            first_try = False
        else:
            cimgs = np.concatenate((cimgs, temp_cimgs), axis=0)
            clabels = np.concatenate((clabels, temp_clabels), axis=0)
    # print(np.unique(cimgs))
    # print(len(cimgs))
    # cimgs_preprocessing = cifar_preprocessing(cimgs)
    # cimgs_preprocessing = cimgs / 255.
    # cimgs_preprocessing = cimgs
    cimgs_preprocessing = svhn_preprocessing(cimgs)
    
    predictions = np.argmax(model.predict(cimgs_preprocessing), axis=1)
    # for idx,img in enumerate(cimgs):
    #     print(clabels[idx], predictions[idx])
    #     print(np.unique(img))
    #     plt.imshow(img, cmap=plt.cm.gray)
    #     plt.show()
    # predictions = np.argmax(model.predict(cimgs), axis=1)
    correct_total = 0
    crash_images = []
    ground_truth = []
    crash_labels = []
    for idx, pred in enumerate(predictions):
        if pred == clabels[idx]:
            correct_total += 1
        else:
            crash_images.append(cimgs[idx])
            ground_truth.append(clabels[idx])
            crash_labels.append(pred)
    print(np.unique(crash_images))
    print("There are %s images that are not crashes. " % correct_total)
    crash_images = np.array(crash_images)
    # crash_iamges_preprocessing = crash_images / 255.0
    # crash_iamges_preprocessing = crash_images
    # crash_iamges_preprocessing = cifar_preprocessing(crash_images)
    crash_iamges_preprocessing = svhn_preprocessing(crash_images)
    ground_truth = np.array(ground_truth)
    crash_labels = np.array(crash_labels)
    correct_total = 0
    predictions = np.argmax(model.predict(crash_iamges_preprocessing), axis=1)
    # predictions = np.argmax(model.predict(crash_images), axis=1)
    for idx, pred in enumerate(predictions):
        if pred == ground_truth[idx]:
            correct_total += 1
    print("After filtering, there are %s images that are not crashes. " % correct_total)
    print("The shape of the crash data: %s", crash_images.shape)
    print("The shape of the ground truth: %s", ground_truth.shape)
    print("The shape of the crash label: %s", crash_labels.shape)
    # assert correct_total == 0
    if os.path.exists(dataset_outdir):
        shutil.rmtree(dataset_outdir)
    os.makedirs(dataset_outdir)
    np.save(os.path.join(dataset_outdir, 'data.npy'), crash_images)
    np.save(os.path.join(dataset_outdir, 'ground_truth.npy'), ground_truth)
    # np.save(os.path.join(dataset_outdir, 'label.npy'), crash_labels)
    # np.save(os.path.join(dataset_outdir, 'original_seed.npy'), original_seed)
    print("successfully saved... ")
    return True

def get_original_seed(crash_output_dir, seed_lists):
    for idx, seed_list in enumerate(seed_lists):
        c_out_dir = os.path.join(crash_output_dir, seed_list)
        imgs, labels = crash_dataset(c_out_dir)
        select_cluster = np.ones(len(labels)) * idx
        if idx == 0:
            original_seed = select_cluster
        else:
            original_seed = np.concatenate((original_seed, select_cluster), axis=0)
    return original_seed

# def image_visualization(imgs, save_dir):
#     # plt.figure(figsize=(20,20))
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for figure_idx in range(len(imgs) // 100 + 1):
#         plt.figure(figsize=(20,20))
#         for image_idx in range(len(imgs)):
#             if 100*figure_idx <= image_idx < 100*(figure_idx+1):
#                 plt.subplot(10, 10, image_idx+1-100*figure_idx)
#                 plt.subplots_adjust(wspace=0.3, hspace=0.3)
#                 plt.axis("off")
#                 # plt.title("%.2f, %.2f" % (temp_tsne[image_idx][0], temp_tsne[image_idx][1]))
#                 plt.imshow(imgs[image_idx], cmap=plt.cm.gray)
#         plt.savefig(os.path.join(save_dir, "%s.png"%figure_idx))
#         plt.cla()
        
if __name__ == '__main__':
    # iters = [500,1000,2000,3000,4000]
    # for i in range(10):
    for i in range(1):
        # my_model_dir = "./new_model/lenet5_softmax.h5"
        my_model_dir = "/data/wlt/cifar10_vgg_model.194.h5"
        
        # my_model_dir = "./deephunter/profile/mnist/models/lenet5.h5"
        # my_model_dir = "./new_model/fm_lenet5.h5"
        crash_dir_list = ['outputs_50']
        crash_output_dir = "./deephunter_outputs/svhn_vgg_ga_nbc_iter_5000_efficient"
        cdataset_out_dir = "/data/wlt/deephunter/deephunter/cdataset_output/svhn_vgg_ga_nbc_iter_5000_efficient"
        if not os.path.exists(cdataset_out_dir):
            os.makedirs(cdataset_out_dir)
        original_seed = get_original_seed(crash_output_dir, crash_dir_list)
        filter_dataset(crash_output_dir, my_model_dir, cdataset_out_dir, crash_dir_list, original_seed)
    # save_dir = os.path.join(cdataset_out_dir, "image_visual")
    # data = np.load(os.path.join(cdataset_out_dir, "data.npy"))
    # image_visualization(data, save_dir)
        