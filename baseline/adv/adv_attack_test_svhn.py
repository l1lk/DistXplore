import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import foolbox
import numpy as np
from collections import Counter
import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Input, AveragePooling2D
from keras import regularizers
from keras.optimizers import SGD, Adam
from scipy import io
import argparse
from get_model import svhn_resnet20, svhn_vgg16



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

def target_adv_attack(tmodel, seeds, labels, target_label, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.models.KerasModel(model=tmodel, bounds=(-2.3, 2.8))
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

def load_svhn():

    x_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['y']

    x_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="truth and target")
    parser.add_argument("-truth", type=int)
    parser.add_argument("-target", type=int)
    args = parser.parse_args()

    model = svhn_vgg16()


    adv_seed_dir = "/data/wlt/distribution-aware-data/seeds/svhn/seed_v2"

    all_class_adv_seeds = []
    for file_index in range(10):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy"%file_index))
        temp_adv_seeds = svhn_preprocessing(temp_adv_seeds)
        # temp_adv_seeds = cifar_preprocessing(temp_adv_seeds)
        all_class_adv_seeds.append(temp_adv_seeds)
    all_class_adv_seeds = np.array(all_class_adv_seeds)    

    for idx, class_data in enumerate(all_class_adv_seeds):
        if idx in [args.truth]:
            label_index = idx - 1
            if label_index == -1:
                label_index = 9
            target_list = np.arange(10)
            target_list = np.delete(target_list, label_index)
            # target_list = [6,7,8,9]
            class_label = np.ones(len(class_data), dtype=np.int32) * int(label_index)
            print("truth: %s"%idx, model.evaluate(class_data, keras.utils.to_categorical(class_label, 10)))
            
            for target_label in [args.target]:
                target_index = target_label - 1
                if target_index == -1:
                    target_index = 9
                index = 0
                for iter in [5, 10]:
                    for eps in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        print(1)
                        adv = target_adv_attack(model, class_data, class_label, target_index, "bim", eps, iter)
                        print(adv.shape)
                        print(class_label.shape)
                        print(label_index, target_index, model.evaluate(adv, keras.utils.to_categorical(class_label,10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir =  "/data/wlt/target_adv_samples/svhn_resnet/bim/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(label_index, target_index,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(label_index, target_index,index)), class_label)
                        index += 1
            
            for target_label in [args.target]:
                target_index = target_label - 1
                if target_index == -1:
                    target_index = 9
                index = 0
                for itera in [5, 10]:
                    for eps in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        adv = target_adv_attack(model, class_data, class_label, target_index, "pgd", eps, itera)
                        print(adv.shape)
                        print(class_label.shape)
                        print("truth:%s"%label_index, "target:%s"%target_index, model.evaluate(adv, keras.utils.to_categorical(class_label,10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/svhn_resnet/pgd/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(label_index, target_index,index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(label_index, target_index,index)), class_label)
                        index += 1
            
            for target_label in [args.target]:
                target_index = target_label - 1
                if target_index == -1:
                    target_index = 9
                index = 0
                for ini_c in [1e-2, 2e-2]:
                    for lr in [5e-3, 6e-3, 7e-3, 8e-3, 9e-3]:
                        print(np.max(class_data))
                        print(np.min(class_data))
                        adv = target_adv_attack(model, class_data, class_label, target_index, "cw", lr, ini_c)
                        print(adv.shape)
                        print(class_label.shape)
                        print(label_index, target_index, model.evaluate(adv, keras.utils.to_categorical(class_label, 10)))
                        print(Counter(np.argmax(model.predict(adv), axis=1)))
                        adv_save_dir = "/data/wlt/target_adv_samples/svhn_resnet/cw/seed_v2"
                        if not os.path.exists(adv_save_dir):
                            os.makedirs(adv_save_dir)
                        np.save(os.path.join(adv_save_dir, "adv_data_class_%s_target_%s_%s.npy"%(label_index, target_index, index)), adv)
                        np.save(os.path.join(adv_save_dir, "adv_label_class_%s_target_%s_%s.npy"%(label_index, target_index, index)), class_label)
                        index += 1