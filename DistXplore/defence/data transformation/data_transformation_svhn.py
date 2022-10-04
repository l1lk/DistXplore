import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD
from scipy import io


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

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

def data_trans_mnist(model, adv_data, adv_label):
    input_height = 32
    input_width = 32
    resize_height = 31
    resize_width = 31
    # adv_data = cifar_preprocessing(adv_data)

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

    pad_test_data = pad_test_data.reshape(-1,32,32,3)
    # print(np.unique(pad_test_data))
    acc = model.evaluate(pad_test_data, keras.utils.to_categorical(adv_label, 10))[1]
    return acc


# model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/cifar10_vgg_model.194.h5")
model = get_svhn_model() 
model.load_weights("../svhn_vgg16_weight.h5")
sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tech_average_acc = []
for tech in ["mmd", "bim", "pgd", "cw"]:
    truth_acc = []
    for truth in range(10):
        target_list = np.arange(10)
        target_list = np.delete(target_list, truth)
        target_acc = []
        for target in target_list:
            # adv_data = np.load("D:/distribution-aware-data/defesnse crashes/svhn/%s/seed_v1/data_%s_%s.npy"%(tech, truth, target))
            # adv_label = np.load("D:/distribution-aware-data/defesnse crashes/svhn/%s/seed_v1/ground_truth_%s_%s.npy"%(tech, truth, target))
            adv_data = np.load("D:/distribution-aware-data/defesnse crashes/svhn/kmnc/data.npy")
            adv_label = np.load("D:/distribution-aware-data/defesnse crashes/svhn/kmnc/ground_truth.npy")
            
            if tech == 'mmd':
                adv_data = svhn_preprocessing(adv_data)
                shuffle_index = np.random.permutation(len(adv_data))
                adv_data = adv_data[shuffle_index]
                adv_label = adv_label[shuffle_index]
                adv_data = adv_data[:1000]
                adv_label = adv_label[:1000]
            # adv_data = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/data.npy")
            # adv_label = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/ground_truth.npy")
            print(adv_data.shape)
            # print(cifar_preprocessing(adv_data).shape)
            # print(model.evaluate(cifar_preprocessing(adv_data), keras.utils.to_categorical(adv_label,10)))
            acc = data_trans_mnist(model, adv_data, adv_label)
            print(truth, target, acc)
            target_acc.append(acc)
            break
        truth_acc.append(target_acc)
        break        
    tech_average_acc.append(np.average(truth_acc))
    break
print(tech_average_acc)
# df = pd.DataFrame(truth_acc)
# df.to_excel("./cw_cifar_acc.xlsx", sheet_name="0")




