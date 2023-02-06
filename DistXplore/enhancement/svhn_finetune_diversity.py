import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import keras
import argparse
import tensorflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import argparse
from scipy import io
import get_model

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

def load_svhn():

    x_train = io.loadmat('/home/dltest/tianmeng/distribution/dataset/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/home/dltest/tianmeng/distribution/dataset/train_32x32.mat')['y']

    x_test = io.loadmat('/home/dltest/tianmeng/distribution/dataset/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/home/dltest/tianmeng/distribution/dataset/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    print(x_train.shape)
    # print(y_train)
    # grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    # grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    # return grouped_x_train, grouped_x_test, y_train
    return (x_train, y_train), (x_test, y_test)

def get_retrain_dataset(simages, struths):
    (X_train, y_train), (X_test, y_test) = load_svhn()
    y_train = y_train.reshape(-1)
    # X_train = svhn_preprocessing(X_train)
    y_train = y_train.astype(np.float32)
    y_train = y_train - 1
    # y_train[np.where(y_train==-1)[0]]=9
    # X_train = X_train.reshape(-1, 28, 28, 1)
    # X_train = X_train / 255.
    X_retrain = np.concatenate((X_train, simages))
    y_retrain = np.concatenate((y_train, struths))
    shuffle_index = np.random.permutation(np.arange(len(X_retrain)))
    X_retrain_shuffled = X_retrain[shuffle_index, ...]
    y_retrain_shuffled = y_retrain[shuffle_index]
    return X_retrain_shuffled, y_retrain_shuffled

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fine tune')
    parser.add_argument('-ft_epoch', type=int)
    args = parser.parse_args()
    
    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False
            )
    
    model_save_dir = "/data/c/tianmeng/wlt/retrain_models/svhn_resnet"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    (X_train, y_train), (X_test, y_test) = load_svhn()

    # X_test = X_test.reshape(-1,28,28,1)
    # X_test = X_test / 255
    # X_test = cifar_preprocessing(X_test)
    y_train = y_train.reshape(-1)
    y_train = y_train.astype(np.float32)
    y_train = y_train - 1
    y_test = y_test.reshape(-1)
    X_test = svhn_preprocessing(X_test)
    y_test = y_test.astype(np.float32)
    # print(np.unique(y_test))
    y_test= y_test - 1
    # print()
    # y_test[np.where(y_test==-1)[0]]=9
    y_test = keras.utils.to_categorical(y_test)
    
    # origin_model = keras.models.load_model("/data/wlt/cifar10_vgg_model.194.h5")
    # origin_model = get_svhn_model()
    for adv_tech in ["bim", "pgd", "cw"]:
        origin_model = get_model.svhn_resnet20()
     
        select_data = np.load("/data/c/dh_retrain_data/svhn_kmnc_data.npy")
        select_truth = np.load("/data/c/dh_retrain_data/svhn_kmnc_truth.npy")
        select_data = X_train
        select_truth = y_train
    
        select_truth = select_truth.astype(np.int32)
        np.random.seed(2)
        shuffle_index = np.random.permutation(len(select_data))
        select_data = select_data[shuffle_index]
        select_truth = select_truth[shuffle_index]
        select_data = select_data[:18900]
        select_truth = select_truth[:18900]

        print(select_data.shape)
        print(np.unique(select_truth))
        print(origin_model.evaluate(svhn_preprocessing(select_data), keras.utils.to_categorical(select_truth,10)))
        retrain_data, retrain_truth = get_retrain_dataset(select_data, select_truth)
        # retrain_data = retrain_data / 255.
        print(np.unique(retrain_data))
        retrain_data = svhn_preprocessing(retrain_data)

        retrain_truth = keras.utils.to_categorical(retrain_truth)
        print(np.unique(retrain_data))
        print(retrain_data.shape)
        print(retrain_truth.shape)
        optimizer = tensorflow.keras.optimizers.SGD(learning_rate=2e-4, decay=1e-6, momentum=0.9, nesterov=True)
        origin_model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        origin_model.fit_generator(datagen.flow(retrain_data, retrain_truth,batch_size=128),
                                                    steps_per_epoch = retrain_data.shape[0]//128,
                                                    epochs=args.ft_epoch,
                                                    validation_data=(X_test, y_test))
        origin_model.save(os.path.join(model_save_dir, "vae_ft_model_%s_all.h5"%args.ft_epoch))
        break

    
