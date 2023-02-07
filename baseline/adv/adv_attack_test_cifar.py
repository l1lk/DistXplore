import numpy as np
from sklearn.mixture import GaussianMixture
from keras.datasets import mnist
import collections
# from tqdm import tqdm
import random
import os
import glob
# import cal_mmd as mmd
import argparse, pickle
import foolbox
import numpy as np
# import torchvision.models as models
from collections import Counter
# import joblib
# import keras
import keras
# import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Input, AveragePooling2D
from keras import regularizers
from keras.optimizers import SGD, Adam
from get_model import cifar_resnet20

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, 
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 40)')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block==0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
        
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, 
                    kernel_initializer='he_normal')(y)
    outputs = Activation('softmax')(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

# for foolbox
def adv_attack(tmodel, seeds, labels, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.models.KerasModel(model=tmodel, bounds=(0,1))
    distance = foolbox.distances.Linfinity

    if method == "fgsm":
        attack = foolbox.attacks.FGSM(fmodel, distance=distance)
        adversarials = attack(seeds, labels, epsilon=1)
        return adversarials
    elif method == "pgd":
        attack = foolbox.attacks.PGD(fmodel, distance=distance)
        adversarials = attack(seeds, labels, epsilon=para_0, iterations=para_1)
        return adversarials
    elif method == "cw":
        attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        adversarials = attack(seeds, labels, learning_rate=para_0, initial_const=para_1)
        return adversarials
    
def target_adv_attack(tmodel, seeds, labels, target_label, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.models.KerasModel(model=tmodel, bounds=(-2, 2.3))
    distance = foolbox.distances.Linfinity
    criteria = foolbox.criteria.TargetClass
    criteria = criteria(target_label)
    
    if method == "bim":
        attack = foolbox.attacks.L2BasicIterativeAttack(fmodel, criterion=criteria)
        adversarials = attack(seeds, labels, epsilon=para_0, iterations=para_1)
        return adversarials
    elif method == "pgd":
        attack = foolbox.attacks.PGD(fmodel, distance=distance, criterion=criteria)
        adversarials = attack(seeds, labels, epsilon=para_0, iterations=para_1)
        return adversarials
    elif method == "cw":
        attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance, criterion=criteria)
        adversarials = attack(seeds, labels, learning_rate=para_0, initial_const=para_1)
        return adversarials



# adv_mode_list = ['pgd']
# for adv_mode in adv_mode_list:
#     model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/lenet5_softmax.h5")
#     # raw_seeds_dir = "D:/Deephunter-backup-backup/test_seeds/distribution_select"
#     # org_seeds = []
#     # org_seeds_label = []
#     # for file in os.listdir(raw_seeds_dir):
#         # org_seeds.append(np.load(os.path.join(raw_seeds_dir, file))[0])
#         # org_seeds_label.append(int(file.split('_')[0]))
#     # org_seeds = np.array(org_seeds)
#     org_seeds = np.load("./mmd_adv_seeds/data.npy")
#     # org_seeds_label = np.array(org_seeds_label)
#     org_seeds_label = np.load("./mmd_adv_seeds/ground_truth.npy")
#     print(org_seeds.shape)
#     print(org_seeds_label.shape)
#     org_seeds_pre = org_seeds / 255
#     # np.save("./data/seed_data/mnist/data.npy", org_seeds_pre)
#     # np.save("./data/seed_data/mnist/ground_truh.npy", org_seeds_label)
    
#     print(model.evaluate(org_seeds_pre, org_seeds_label))
#     # test_seed = np.load("../seeds/mnist_test/class_0_1_seed.npy")
#     # test_seed = test_seed.reshape(-1,28,28,1)
#     # test_seed = test_seed / 255
#     # print(np.unique(test_seed))
#     # print(test_seed.shape)
#     # label = np.array([0])
#     index = 0
#     for ini_c in [300, 400]:
#         for lr in [0.1, 0.3, 0.5, 0.7, 0.9]:
#             adv = target_adv_attack(model, org_seeds_pre, org_seeds_label, 0, adv_mode, lr=lr, ini_const=ini_c)
#             print(model.evaluate(adv, org_seeds_label))
#             print(Counter(np.argmax(model.predict(adv), axis=1)))
#             adv_save_dir = os.path.join("./data/adv_data/mnist", adv_mode)
#             # if not os.path.exists(adv_save_dir):
#                 # os.makedirs(adv_save_dir)
#             # np.save(os.path.join(adv_save_dir, "adv_data_%s.npy"%index), adv)
#             # np.save(os.path.join(adv_save_dir, "adv_label_%s.npy"%index), org_seeds_label)
#             # index += 1
# # print(np.unique(adv))
# # print(np.argmax(model.predict(test_seed), axis=1))
# # print(np.argmax(model.predict(adv), axis=1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="truth and target")
    parser.add_argument("-truth", type=int)
    parser.add_argument("-target", type=int)
    args = parser.parse_args()
    model = keras.models.load_model("/data/wlt/cifar10_vgg_model.194.h5")
    # model = cifar_resnet20()
    # model = resnet_v1(input_shape=(32,32,3), depth=20)
    # model.load_weights("/data/wlt/resnet20_cifar_weights.h5")
    # model.compile(loss='categorical_crossentropy',
                # optimizer=Adam(lr=1e-3),
                # metrics=['accuracy'])
    # model = keras.models.load_model("./models/lenet5_softmax.h5")
    # model = keras.models.load_model("/data/wlt/fm_lenet5.h5")

    adv_seed_dir = "/data/wlt/distribution-aware-data/seeds/cifar10/seed_v2"
    # adv_seed_dir = "/data/wlt/training_100_v2/training_100_v2"
    # adv_seed_dir = "/data/wlt/distribution-aware-data/seeds/fmnist/seed_v1"
    all_class_adv_seeds = []
    for file_index in range(10):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy"%file_index))
        temp_adv_seeds = cifar_preprocessing(temp_adv_seeds)
        # temp_adv_seeds = temp_adv_seeds.reshape(-1, 28, 28, 1)
        # temp_adv_seeds = temp_adv_seeds / 255.
        all_class_adv_seeds.append(temp_adv_seeds)
    all_class_adv_seeds = np.array(all_class_adv_seeds)    
    # print(all_class_adv_seeds.shape)
    # (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # X_train = X_train.reshape(-1,28,28,1)
    # X_train = X_train / 255
    # X_test = X_test.reshape(-1,28,28,1)
    # X_test = X_test / 255
    # print(model.evaluate(X_train, y_train))
    # print(model.evaluate(X_test, y_test))
    for idx, class_data in enumerate(all_class_adv_seeds):
        if idx in [args.truth]:
            # for index, data in enumerate(class_data):
            #     plt.imshow(data)
            #     plt.savefig("./images_show/%s_%s.png"%(idx,index))
            target_list = np.arange(10)
            target_list = np.delete(target_list, idx)
            class_label = np.ones(len(class_data), dtype=np.int32) * int(idx)
            # class_label = keras.utils.to_categorical(class_label, 10)
            # class_data = class_data.reshape(-1,28,28,1)
            # print(np.unique(class_data))
            # class_data_pre = class_data / 255.
            # print(np.unique(class_data_pre))
            print(idx, model.evaluate(class_data, keras.utils.to_categorical(class_label, 10)))
            # print(idx, model.evaluate(class_data, class_label))

            for target_label in [args.target]:
                index = 0
                for iter in [5, 10]:
                    for eps in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        adv = target_adv_attack(model, class_data, class_label, target_label, "bim", eps, iter)
                        print(adv.shape)
                        print(class_label.shape)
                        print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        # print(idx, target_label, model.evaluate(adv, class_label))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/cifar_resnet/bim/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(idx, target_label,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(idx, target_label,index)), class_label)
                        index += 1
            
            for target_label in [args.target]:
                index = 0
                for itera in [5, 10]:
                    for eps in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        adv = target_adv_attack(model, class_data, class_label, target_label, "pgd", eps, itera)
                        print(adv.shape)
                        print(class_label.shape)
                        print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label,10)))
                        # print(idx, target_label, model.evaluate(adv, class_label))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/cifar_resnet/pgd/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(idx, target_label,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(idx, target_label,index)), class_label)
                        index += 1

            for target_label in [args.target]:
                index = 0
                for ini_c in [1e-2, 2e-2]:
                    for lr in [5e-3, 6e-3, 7e-3, 8e-3, 9e-3]:
                        print(np.max(class_data))
                        print(np.min(class_data))
                        adv = target_adv_attack(model, class_data, class_label, target_label, "cw", lr, ini_c)
                        print(adv.shape)
                        print(class_label.shape)
                        print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/cifar_resnet/cw/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(idx, target_label,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(idx, target_label,index)), class_label)
                        index += 1