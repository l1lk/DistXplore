import numpy as np

import os
import foolbox
import numpy as np
from collections import Counter
# from tensorflow import keras

import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Input, AveragePooling2D, MaxPool2D
from keras import regularizers
from keras.optimizers import SGD, Adam
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def get_lenet4_model():
    model = Sequential()
    model.add(Conv2D(4,input_shape=(28,28,1),kernel_size=(5,5),padding='valid',strides=(1,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(6,kernel_size=(5,5),padding='valid',strides=(1,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dense(84,activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10)) # logits
    model.add(Activation('softmax')) # softmax
    return model


def target_adv_attack(tmodel, seeds, labels, target_label, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.models.KerasModel(model=tmodel, bounds=(0, 1))
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



# model = keras.models.load_model("/data/wlt/cifar10_vgg_model.194.h5")
# model = keras.models.load_model("/data/wlt/lenet5_softmax.h5")
# model = keras.models.load_model("/data/wlt/fm_lenet5.h5")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="truth and target")
    parser.add_argument("-truth", type=int)
    parser.add_argument("-target", type=int)
    args = parser.parse_args()
    model = get_lenet4_model()
    model.load_weights("/data/wlt/lenet4_mnist_weights.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # adv_seed_dir = "/data/wlt/single_cluster_seeds/cifar10_training_100/training_100"
    adv_seed_dir = "/data/wlt/distribution-aware-data/seeds/mnist/seed_v2"
    all_class_adv_seeds = []
    for file_index in range(10):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy"%file_index))
        # temp_adv_seeds = cifar_preprocessing(temp_adv_seeds)
        temp_adv_seeds = temp_adv_seeds.reshape(-1, 28, 28, 1)
        temp_adv_seeds = temp_adv_seeds / 255.
        all_class_adv_seeds.append(temp_adv_seeds)
    all_class_adv_seeds = np.array(all_class_adv_seeds)    
    for idx, class_data in enumerate(all_class_adv_seeds):
        if idx in [args.truth]:
            target_list = np.arange(10)
            target_list = np.delete(target_list, idx)
            target_list = [1]
            class_label = np.ones(len(class_data), dtype=np.int32) * int(idx)
            print(idx, model.evaluate(class_data, keras.utils.to_categorical(class_label, 10)))
            # print(idx, model.evaluate(class_data, class_label))

            for target_label in [args.target]:
                index = 0
                for iter in [5, 10]:
                    for eps in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        adv = target_adv_attack(model, class_data, class_label, target_label, "bim", eps, iter)
                        print(adv.shape)
                        print(class_label.shape)
                        # print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/mnist_lenet4/bim/seed_v2"
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
                        # print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label,10)))
                        # print(idx, target_label, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/mnist_lenet4/pgd/seed_v2"
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
                        adv_save_dir = "/data/wlt/target_adv_samples/mnist_lenet4/cw/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(idx, target_label,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(idx, target_label,index)), class_label)
                        index += 1
    