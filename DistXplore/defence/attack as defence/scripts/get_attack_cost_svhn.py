from cmath import isnan
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import foolbox
from progress.bar import Bar
import math

import attacks_method.saliency as saliency
from attacks_method.iterative_projected_gradient import BIM, L2BasicIterativeAttack
import attacks_method.boundary_attack as DBA
import gl_var
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD
import argparse
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


def attack_for_input(model, x_input, y_input, method, dataset, save_path):
    """Gen attack costs for input data by input attack method

    Args:
        model: keras model
        input_data [str]: input data name
        method [str]: attack method for defense

    """  
    # * attack parameters
    if dataset == 'mnist':
        img_rows, img_cols, img_rgb = 28, 28, 1
        epsilon = 0.3
        stepsize = 0.0006
        epsilon_l2 = 0.1
        stepsize_l2 = 0.0002
    elif dataset == 'cifar':
        img_rows, img_cols, img_rgb = 32, 32, 3
        epsilon = 0.03
        stepsize = 0.00006
        epsilon_l2 = 0.01
        stepsize_l2 = 0.00002
    else:
        raise Exception('Input dataset error:', dataset)
    fmodel = foolbox.models.KerasModel(model, bounds=(-2.3, 2.8))

    f = open(save_path, 'w')

    # check the input bound
    # assert np.min(x_input) == 0 and np.max(x_input) == 1
    x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, img_rgb).astype('float')
    loss = model.evaluate(x_input, keras.utils.to_categorical(y_input, 10))[0]
    # loss = model.evaluate(x_input, y_input)[0]
    print(loss)
    if math.isnan(loss):
        print("close")
        return None
    bar = Bar('Attack processing', max=x_input.shape[0])
    for index, data in enumerate(x_input):
        # if np.min(data)<0 or np.max(data)>1:
        #     continue
        fmodel_predict = fmodel.predictions(data)
        label = np.argmax(fmodel_predict)

        # define criterion
        if method != 'JSMA':
            # untarged attack
            criterion = foolbox.criteria.Misclassification()
        else:
            # JSMA attack only support targeted attack, second target attack
            second_result = fmodel_predict.copy()
            second_result[np.argmax(second_result)] = np.min(second_result)
            target_label = np.argmax(second_result)
            criterion = foolbox.criteria.TargetClass(int(target_label))

        # define method
        if method == 'BIM':
            gl_var.return_iter_bim = -500
            attack = BIM(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)
            adversarial = attack(data, label, binary_search=False,
                                 epsilon=epsilon, stepsize=stepsize, iterations=500)
            attack_iter = gl_var.return_iter_bim
            max_iter = 500

        elif method == 'BIM2':
            gl_var.return_iter_bim = -500
            attack = L2BasicIterativeAttack(fmodel, criterion=criterion)  # distance=foolbox.distances.Linfinity
            adversarial = attack(data, label, binary_search=False,
                                 epsilon=epsilon_l2, stepsize=stepsize_l2, iterations=500)
            attack_iter = gl_var.return_iter_bim
            max_iter = 500

        elif method == 'JSMA':
            attack = saliency.SaliencyMapAttack(fmodel, criterion)
            gl_var.return_iter_jsma = -2000
            # print(np.unique(data))
            adversarial = attack(data, label)
            attack_iter = gl_var.return_iter_jsma
            max_iter = 2000

        elif method == 'DBA':
            gl_var.return_iter_dba = -5000
            attack = DBA.BoundaryAttack(fmodel, criterion=criterion)
            adversarial = attack(data, label)
            attack_iter = gl_var.return_iter_dba
            max_iter = 5000

        else:
            raise Exception('Invalid attack method:', method)

        # * adversarial confirm part
        # * as we use foolbox to attack, ignore this part
        # adv_predict = fmodel.predictions(adversarial)
        # adv_label = np.argmax(fmodel_predict)
        # assert adv_label != label

        if adversarial is not None:
            f.write(str(attack_iter) + "\n")
        # adversarial is none means the attack method cannot find an adv, so return the max iterations
        else:
            f.write(str(max_iter) + "\n")
        
        bar.suffix = '({index}/{size}) | Total: {total:} | ETA: {eta:}'.format(
                    index=index+1,
                    size=x_input.shape[0],
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish()
    f.close() # close the file


def generate_attack_cost(kmodel, x_input, y_input, dataset, attack_method, gtruth, target):
    """ 
    Generate attack costs for benign and adversarial examples.
    """  
    
    # x_input, y_input, _, _ = get_data(dataset) # the input data min-max -> 0.0 1.0
    
    # except wrong label data
    # preds_test = kmodel.predict_classes(x_input)
    preds_test = np.argmax(kmodel.predict(x_input), axis=1)
    # inds_correct = np.where(preds_test == y_input)[0]
    # x_input = x_input[inds_correct]
    # x_input = x_input[0: 1000]
    # save_path = '../results/detector/' + dataset + '_softmax_mmd_%s_%s_%s'%(truth, target, iter)
    # save_path = '../results/detector/cifar_softmax_mmd_%s_%s' % (gtruth, target)
    save_path = '../results/detector/svhn_resnet20_pgd_%s_%s'%(gtruth, target)
    save_path = "../results/detector/svhn_resnet_nbc"
    # save_path = '../results/detector/' + dataset + '_softmax_nbc'
    attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)

# model = keras.models.load_model("../data/lenet5_softmax.h5")        
# model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/cifar10_vgg_model.194.h5")
# model = keras.models.load_model("D:/Deephunter-backup-backup/deephunter/new_model/cifar10_vgg_model.194.h5")
# model = keras.models.load_model("D:/retrain_model/fm_lenet5.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
    parser.add_argument('-select_truth', help="realistic transformation type", type=int)
    args = parser.parse_args()
    # model = get_svhn_model()
    # model.load_weights("../svhn_vgg16_weight.h5")
    # (X_train,_), _ = keras.datasets.cifar10.load_data()
    # sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model = keras.models.load_model("/data/wlt/cifar10_vgg_model.194.h5")
    model = get_model.svhn_resnet20()
    # for iteration in range(1,32):
    for gtruth in [args.select_truth]:
        target_list = np.arange(10)
        target_list = np.delete(target_list, gtruth)
        for target in target_list:
            # x_adv = np.load("../all_mmd_adv_iter/data_%s_%s_%s.npy"%(gtruth, target, iter))
            # print(x_adv.shape)
            # y_adv = np.load("../all_mmd_adv_iter/ground_truth_%s_%s_%s.npy"%(gtruth, target, iter))
            x_adv = np.load("/data/c/all_adv_data/svhn_resnet/pgd/data_%s_%s.npy"%(gtruth, target))
            y_adv = np.load("/data/c/all_adv_data/svhn_resnet/pgd/ground_truth_%s_%s.npy"%(gtruth, target))

            # x_adv = np.load("/data/c/distribution-aware-data/defesnse crashes/cifar10/hda/data.npy")
            # y_adv = np.load("/data/c/distribution-aware-data/defesnse crashes/cifar10/hda/ground_truth.npy")
            shuffle_index = np.random.permutation(len(x_adv))
            x_adv = x_adv[shuffle_index]
            y_adv = y_adv[shuffle_index]
            x_adv = x_adv[:1000]
            y_adv = y_adv[:1000]
            # x_adv = np.load("D:/all_adv_data/cifar_cw_3/data_%s_%s.npy"%(gtruth, target))
            # y_adv = np.load("D:/all_adv_data/cifar_cw_3/ground_truth_%s_%s.npy"%(gtruth, target))
            # x_adv = np.load("../ga_fm_crash/data_%s_%s.npy"%(gtruth, target))
            # y_adv = np.load("../ga_fm_crash/ground_truth_%s_%s.npy"%(gtruth, target))
            # x_adv = np.load("./mnist_distribution_1/mnist_aes_seed_0/class_0/truth_0_0.npy")
            # y_adv = np.ones(len(x_adv), dtype=np.int32) * 0
            print(x_adv.shape)
            print(np.unique(x_adv))
            # x_adv = np.load("../mmd_ga_mnist_iteration_2/data_0_1_%s.npy"%iteration)
            # y_adv = np.load("../mmd_ga_mnist_iteration_2/ground_truth_0_1_%s.npy"%iteration)
            # print(type(x_adv))
            # x_adv = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/data.npy")
            # y_adv = np.load("../../vgg_mmd_ga_nbc_iter_5000_0/ground_truth.npy")
            # print(model.evaluate(x_adv / 255., y_adv))
            # shuffle_index = np.random.permutation(len(x_adv))
            # x_adv = x_adv[shuffle_index]
            # y_adv = y_adv[shuffle_index]
            # x_adv = x_adv[:300]
            # y_adv = y_adv[:300]
            
            # x_adv = cifar_preprocessing(x_adv)
            x_adv = svhn_preprocessing(x_adv)
            # x_adv = x_adv / 255.
            # print(np.max(x_adv))
            # print(np.min(x_adv))
            # x_adv = x_adv / 255.
            # print(np.unique(x_adv))
            # (x_train, y_train), _ = keras.datasets.mnist.load_data()
            # x_train = x_train[:1000]
            # y_train = y_train[:1000]
            # x_train = x_train.reshape(-1,28,28,1)
            # x_train = x_train / 255.

            generate_attack_cost(model, x_adv, y_adv, 'cifar', "JSMA", gtruth, target)

            

