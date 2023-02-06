import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import keras
import tensorflow
import argparse
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import get_model



def get_retrain_dataset(simages, struths):
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    y_train = y_train.reshape(-1)
    # X_train = cifar_preprocessing(X_train)
    X_train = X_train.reshape(-1, 28, 28, 1)
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
    parser.add_argument('-truth')
    parser.add_argument('-target')
    parser.add_argument('-snum')
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
    
    model_save_dir = "/data/c/tianmeng/wlt/retrain_models/fmnist_lenet4"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1,28,28,1)
    X_test = X_test / 255
    # X_test = cifar_preprocessing(X_test)
    # y_test = keras.utils.to_categorical(y_test)
    
    # origin_model = keras.models.load_model("/data/wlt/cifar10_vgg_model.194.h5")
    # origin_model = keras.models.load_model("/data1/wlt/fm_lenet5.h5")
    for adv_tech in ["bim", "pgd", "cw"]:
        origin_model = get_model.fmnist_lenet4()

        select_data = np.load("/data/c/dh_retrain_data/fmnist_nbc_data.npy")
        select_truth = np.load("/data/c/dh_retrain_data/fmnist_nbc_truth.npy")
        np.random.seed(2)
        shuffle_index = np.random.permutation(len(select_data))
        select_data = select_data[shuffle_index]
        select_truth = select_truth[shuffle_index]
        select_data = select_data[:24300]
        select_truth = select_truth[:24300]
        select_truth = select_truth.astype(np.int32)
        nan_idx_list = []
        print(np.unique(select_data))
        for idx, data in enumerate(select_data):
            if math.isnan(np.unique(data)[-1]):
                nan_idx_list.append(idx)
        select_data = np.delete(select_data, nan_idx_list, axis=0)
        select_truth = np.delete(select_truth, nan_idx_list, axis=0)
        # select_truth = np.ones(len(select_data), dtype=np.int32)*9
        print(select_data.shape)
        # print(np.unique(select_truth))
        retrain_data, retrain_truth = get_retrain_dataset(select_data, select_truth)
        retrain_data = retrain_data / 255.
        # print(np.unique(retrain_data)[:1000])
        # retrain_data = cifar_preprocessing(retrain_data)

        # retrain_truth = keras.utils.to_categorical(retrain_truth)
        print(np.unique(retrain_data))
        print(retrain_data.shape)
        print(retrain_truth.shape)
        optimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)
        origin_model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
                        optimizer=optimizer,
                        metrics=['accuracy'])

        origin_model.fit(retrain_data, retrain_truth, epochs=args.ft_epoch, batch_size=128, verbose=2, validation_data=(X_test, y_test))                                                
        origin_model.save(os.path.join(model_save_dir, "vae_ft_model_%s_all.h5"%args.ft_epoch))
        break
        

    
