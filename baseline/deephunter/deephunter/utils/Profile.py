'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

import pickle
import pprint
from re import L
from PIL import Image

from keras import Model
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import load_model
import numpy as np
import collections

import os, sys, errno
from keras import backend as K
import keras
import os

from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD
from scipy import io
import get_model


def svhn_load_data():
    save_dir = "./svhn_dataset"
    x_train = np.load(os.path.join(save_dir, "x_train.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    x_test = np.load(os.path.join(save_dir, "x_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_train = y_train - 1
    y_train[np.where(y_train==-1)[0]] = 9
    y_test = y_test - 1
    y_test[np.where(y_test==-1)[0]] = 9
    return (x_train, y_train), (x_test, y_test)

def load_svhn():

    x_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['y']

    x_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_train = y_train - 1
    y_test = y_test - 1
    return (x_train, y_train), (x_test, y_test)



class DNNProfile():
    def __init__(self, model, exclude_layer=['input', 'flatten'],
                 only_layer=""):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param model_name: ImageNet Model name, can have ('vgg16','vgg19','resnet50')
        :param neuron_layer: Only these layers are considered for neuron coverage
        '''
        self.model = model
        self.outputs = []

        print('models loaded')

        # the layers that are considered in neuron coverage computation
        self.layer_to_compute = []
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.outputs.append(layer.output)
                self.layer_to_compute.append(layer.name)

        if only_layer != "":
            self.layer_to_compute = [only_layer]

        self.cov_dict = collections.OrderedDict()

        print("* target layer list:", self.layer_to_compute)


        for layer_name in self.layer_to_compute:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                # [mean_value_new, squared_mean_value, standard_deviation, lower_bound, upper_bound]
                self.cov_dict[(layer_name, index)] = [0.0, 0.0, 0.0, None, None]



    def count_layers(self):
        return len(self.layer_to_compute)

    def count_neurons(self):
        return len(self.cov_dict.items())

    def count_paras(self):
        return self.model.count_params()

    def update_coverage(self, input_data):

        inp = self.model.input
        functor = K.function([inp] + [K.learning_phase()], self.outputs)
        outputs = functor([input_data, 0])


        for layer_idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[layer_idx]

            # handle the layer output by each data
            # iter is the number of data
            for iter, layer_output in enumerate(layer_outputs):
                if iter % 1000 == 0:
                    print("*layer {0}, current/total iteration: {1}/{2}".format(layer_idx, iter + 1, len(layer_outputs)))

                for neuron_idx in range(layer_output.shape[-1]):
                    neuron_output = np.mean(layer_output[..., neuron_idx])

                    profile_data_list = self.cov_dict[(layer_name, neuron_idx)]

                    mean_value = profile_data_list[0]
                    squared_mean_value = profile_data_list[1]

                    lower_bound = profile_data_list[3]
                    upper_bound = profile_data_list[4]

                    total_mean_value = mean_value * iter
                    total_squared_mean_value = squared_mean_value * iter

                    mean_value_new = (neuron_output + total_mean_value) / (iter + 1)
                    squared_mean_value = (neuron_output * neuron_output + total_squared_mean_value) / (iter + 1)


                    standard_deviation = np.math.sqrt(abs(squared_mean_value - mean_value_new * mean_value_new))

                    if (lower_bound is None) and (upper_bound is None):
                        lower_bound = neuron_output
                        upper_bound = neuron_output
                    else:
                        if neuron_output < lower_bound:
                            lower_bound = neuron_output

                        if neuron_output > upper_bound:
                            upper_bound = neuron_output

                    profile_data_list[0] = mean_value_new
                    profile_data_list[1] = squared_mean_value
                    profile_data_list[2] = standard_deviation
                    profile_data_list[3] = lower_bound
                    profile_data_list[4] = upper_bound

                    self.cov_dict[(layer_name, neuron_idx)] = profile_data_list



    def dump(self, output_file):

        print("*profiling neuron size:", len(self.cov_dict.items()))
        for item in self.cov_dict.items():
            print(item)
        pickle_out = open(output_file, "wb")
        pickle.dump(self.cov_dict, pickle_out)
        pickle_out.close()

        print("write out profiling coverage results to ", output_file)
        print("done.")


def preprocessing_test_batch(x_test):

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def mnist_preprocessing(x):
    x = x.reshape(x.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(28, 28))
        img = np.asarray(img).astype(np.float32) / 255.0 - 0.1306604762738431
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x

def fashion_mnist_preprocessing(x):
    x = x.reshape(x.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(28, 28))
        img = np.asarray(img).astype(np.float32) / 255
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x


cifar_mean = np.array([125.307, 122.95, 113.865])
cifar_std = np.array([62.9932, 62.0887, 66.7048])
def cifar_preprocessing(x):
    new_x = []
    for img in x:
        img = (img.astype('float32') - cifar_mean) / cifar_std
        new_x.append(img)
    new_x = np.stack(new_x)
    return new_x


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        

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






if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description='neuron output profiling')
    parser.add_argument('-model', help="target model to profile")
    parser.add_argument('-train', help="training data", choices=['mnist', 'cifar', 'fashion_mnist', 'svhn', "mnist_lenet4", "fmnist_lenet4", "cifar_resnet", "svhn_resnet"])
    parser.add_argument('-o', help="output path")
    parser.add_argument('-gpu_index', help='select gpu')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    if args.train == 'svhn':
        model = get_svhn_model()
        model.load_weights("./utils/svhn_vgg16_weight.h5")
        sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    elif args.train == 'mnist_lenet4':
        model = get_model.mnist_lenet4()
    elif args.train == 'fmnist_lenet4':
        model = get_model.fmnist_lenet4()
    elif args.train == 'cifar_resnet':
        model = get_model.cifar_resnet20()
    elif args.train == 'svhn_resnet':
        model = get_model.svhn_resnet20()
    else:
        model = load_model(args.model)
    print('Successfully loaded', model.name)
    model.summary()


    make_sure_path_exists(args.o)

    profiling_dict_result ="{0}.pickle".format(args.o)
    print("profiling output file name {0}".format(profiling_dict_result))


    # get the training data for profiling

    if args.train == 'mnist':
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        x_train = mnist_preprocessing(x_train)
        profiler = DNNProfile(model)
        print(np.shape(x_train))
        profiler.update_coverage(x_train)
        profiler.dump(profiling_dict_result)
    elif args.train == 'fashion_mnist':
        (x_train, train_label), (x_test, test_label) = fashion_mnist.load_data()
        x_train = fashion_mnist_preprocessing(x_train)
        profiler = DNNProfile(model)
        print(np.shape(x_train))
        profiler.update_coverage(x_train)
        profiler.dump(profiling_dict_result)
    elif args.train == 'svhn':
        (x_train, train_label), (x_test, test_label) = svhn_load_data()
        print(x_train.shape)
        profiler = DNNProfile(model)
        for i in range(100):
            x_sub = svhn_preprocessing(x_train[i*600:(i+1)*600])
            profiler.update_coverage(x_sub)
        x_sub = svhn_preprocessing(x_train[60000:])
        profiler.update_coverage(x_sub)
        profiler.dump(profiling_dict_result)
    elif args.train == 'cifar':
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        profiler = DNNProfile(model)
        for i in range(100):
            x_sub = cifar_preprocessing(x_train[i*500:(i+1)*500])
            print(x_sub.shape)
            profiler.update_coverage(x_sub)
        profiler.dump(profiling_dict_result)
    elif args.train == 'mnist_lenet4':
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        x_train = mnist_preprocessing(x_train)
        profiler = DNNProfile(model)
        print(np.shape(x_train))
        profiler.update_coverage(x_train)
        profiler.dump(profiling_dict_result)
    elif args.train == 'fmnist_lenet4':
        (x_train, train_label), (x_test, test_label) = fashion_mnist.load_data()
        x_train = fashion_mnist_preprocessing(x_train)
        profiler = DNNProfile(model)
        print(np.shape(x_train))
        profiler.update_coverage(x_train)
        profiler.dump(profiling_dict_result)
    elif args.train == 'cifar_resnet':
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        profiler = DNNProfile(model)
        for i in range(100):
            x_sub = cifar_preprocessing(x_train[i*500:(i+1)*500])
            print(x_sub.shape)
            profiler.update_coverage(x_sub)
        profiler.dump(profiling_dict_result)
    elif args.train == 'svhn_resnet':
        (x_train, train_label), (x_test, test_label) = load_svhn()
        print(model.evaluate(svhn_preprocessing(x_train), keras.utils.to_categorical(train_label, 10)))
        print(model.evaluate(svhn_preprocessing(x_test), keras.utils.to_categorical(test_label, 10)))
        print(x_train.shape)
        profiler = DNNProfile(model)
        for i in range(100):
            x_sub = svhn_preprocessing(x_train[i*600:(i+1)*600])
            profiler.update_coverage(x_sub)
        x_sub = svhn_preprocessing(x_train[60000:])
        profiler.update_coverage(x_sub)
        profiler.dump(profiling_dict_result)
    else:
        print('Please extend the new train data here!')
