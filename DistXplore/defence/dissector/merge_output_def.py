import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Input, AveragePooling2D, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam

from scipy import io
from sklearn.metrics import roc_curve, auc


def load_svhn():

    x_train = io.loadmat('/home/distribution/dataset/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/home/distribution/dataset/train_32x32.mat')['y']

    x_test = io.loadmat('/home/distribution/dataset/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/home/distribution/dataset/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)


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


def dissector(data, truth, dataset, model_type, preprocess=True):
    """training data"""
    if dataset=="mnist":
        (X_train, y_train), _ = keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.
        if preprocess:
            data = data / 255.
    elif dataset=="fmnist":
        (X_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.
        if preprocess:
            data = data / 255.
    elif dataset=="cifar":
        (X_train, y_train), _ = keras.datasets.cifar10.load_data()
        X_train = cifar_preprocessing(X_train)
        if preprocess:
            data = cifar_preprocessing(data)
    elif dataset=="svhn":
        (X_train, y_train), _ = load_svhn()
        X_train = svhn_preprocessing(X_train)
        y_train = y_train.astype(np.int32)
        y_train = y_train - 1
        if preprocess:
            data = svhn_preprocessing(data)
    
    if model_type=="lenet4":
        model = get_lenet4_model()
        model.load_weights("/data/c/lenet4_%s_weights.h5"%dataset)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        y_train = keras.utils.to_categorical(y_train, 10)
        truth = keras.utils.to_categorical(truth, 10)
        sub_model_dir = "/data/c/sub_models/%s_lenet4"%dataset
    elif model_type=="lenet5":
        if dataset=="mnist":
            model = keras.models.load_model("/data/c/lenet5_softmax.h5")
        elif dataset=="fmnist":
            model = keras.models.load_model("/data/c/fm_lenet5.h5")
        sub_model_dir = "/data/c/tianmeng/wlt/sub_models/%s_lenet5"%dataset
    elif model_type=="vgg16":
        if dataset=="cifar":
            model = keras.models.load_model("/data/c/cifar10_vgg_model.194.h5")
        elif dataset=="svhn":
            model = get_svhn_model() 
            model.load_weights("/data/c/svhn_vgg16_weight.h5")
            sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        y_train = keras.utils.to_categorical(y_train, 10)
        truth = keras.utils.to_categorical(truth, 10)
        sub_model_dir = "/data/c/sub_models/%s_vgg16"%dataset
    elif model_type=="resnet20":
        model = resnet_v1(input_shape=(32,32,3), depth=20)
        model.load_weights("/data/c/resnet20_%s_weights.h5"%dataset)
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=1e-3),
                    metrics=['accuracy'])
        y_train = keras.utils.to_categorical(y_train, 10)
        truth = keras.utils.to_categorical(truth, 10)
        sub_model_dir = "/data/c/sub_models/%s_resnet20"%dataset
    print(X_train.shape)
    neg_data = X_train[np.where(y_train==np.unique(truth)[0])[0]]
    neg_truth = y_train[np.where(y_train==np.unique(truth)[0])[0]]
    neg_data = neg_data[:1000]
    neg_truth = neg_truth[:1000]
    auc_neg_list = np.zeros(len(neg_data), dtype=np.int32)        
    # print(neg_truth.shape)
    auc_pos_list = np.ones(len(data), dtype=np.int32)
    auc_truth_list = np.concatenate((auc_pos_list, auc_neg_list), axis=0)
    data = np.concatenate((data, neg_data), axis=0)
    # print(np.unique(data))
    truth = np.concatenate((truth, neg_truth),axis=0)
    snap_shot = []
    ori_predict_label = np.argmax(model.predict(data), axis=1)
    
    # sub_model_dir = "./models/fmnist"
    for file in os.listdir(sub_model_dir):
        # print(file)
        submodel = keras.models.load_model(os.path.join(sub_model_dir, file))
        snap_shot.append(submodel.predict(data))
    snap_shot.append(model.predict(data))
    snap_shot = np.array(snap_shot)
    all_sample_sv_score = []
    for i in range(snap_shot.shape[1]):
        target_snapshot = snap_shot[:,i,:]
        predict_label = ori_predict_label[i]
        sv_score_list = []
        for ss in target_snapshot:
            if np.argmax(ss)==predict_label:
                l_x = np.argsort(ss)[-1]
                l_sh = np.argsort(ss)[-2]
                l_x_value = np.sort(ss)[-1]
                l_sh_value = np.sort(ss)[-2]
                sv_value = l_x_value / (l_x_value + l_sh_value)
            elif np.argmax(ss)!=predict_label:
                l_H = np.argsort(ss)[-1]
                l_x = predict_label
                l_H_value = np.sort(ss)[-1]
                l_x_value = ss[predict_label]
                sv_value = 1 - (l_H_value / (l_x_value + l_H_value))
            sv_score_list.append(sv_value)
        all_sample_sv_score.append(sv_score_list)
    all_sample_sv_score = np.array(all_sample_sv_score)

    all_sample_final_score = []
    for sv_score in all_sample_sv_score:
        final_score = 0.
        all_weights = 0.
        for idx, score in enumerate(sv_score):
            final_score += score * (idx + 1)
            all_weights += (idx+1)
        # print(all_weights)
        all_sample_final_score.append(final_score / all_weights)
    auc_score_list = []
    for sample_final_score in all_sample_final_score:
        auc_score_list.append([sample_final_score, 1 - sample_final_score])
    print(auc_truth_list)
    fpr, tpr, thres = roc_curve(auc_truth_list, all_sample_final_score, pos_label=0)
    # auc_score = roc_auc_score(auc_truth_list, auc_score_list)
    auc_score = auc(fpr, tpr)
    return auc_score
 
    

if __name__ == "__main__":
    data = 123
    truth = 321
    # dissector(data, truth, "mnist", "lenet4")
    # dissector(data, truth, "mnist", "lenet5")
    # dissector(data, truth, "fmnist", "lenet4")
    # dissector(data, truth, "fmnist", "lenet5")
    # dissector(data, truth, "cifar", "vgg16")
    # dissector(data, truth, "cifar", "resnet20")
    # dissector(data, truth, "svhn", "vgg16")
    # dissector(data, truth, "svhn", "resnet20")
    
