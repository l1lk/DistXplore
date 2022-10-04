import sys
sys.path.append('../')
import numpy as np
import foolbox
import os
import argparse
from keras.models import load_model
import keras
import scipy
from detect.util import get_data
from utils import attack_for_input, get_single_detector_train_data
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD


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

def load_svhn():

    x_train = scipy.io.loadmat('../svhn/train_32x32.mat')['X'] # 73257
    y_train = scipy.io.loadmat('../svhn/train_32x32.mat')['y']

    x_test = scipy.io.loadmat('../svhn/test_32x32.mat')['X'] # 26032 
    y_test = scipy.io.loadmat('../svhn/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    print(x_train.shape)
    # print(y_train)
    # grouped_x_train = mmdcov.group_into_class(x_train, y_train)
    # grouped_x_test = mmdcov.group_into_class(x_test, y_test)

    # return grouped_x_train, grouped_x_test, y_train
    return (x_train, y_train), (x_test, y_test)



def generate_training_data(model, dataset):
    """ 
    Generate adversarial examples for training,
    as we use the test set to calc auroc, 
    now we use the training set to generate new adversarial examples.
    """  

    # use training set to generate adv
    # x_input, y_input, _, _ = get_data('mnist') # the input data min-max -> 0.0 1.0
    # (x_input, y_input), (_, _) = keras.datasets.cifar10.load_data()
    # x_input = cifar_preprocessing(x_input)
    # y_input = y_input.reshape(-1)
    (x_input, y_input), _ = load_svhn()
    x_input = svhn_preprocessing(x_input)
    y_input = y_input.reshape(-1)
    y_input = y_input - 1
    # print("acc", model.evaluate(x_input, keras.utils.to_categorical(y_input, 10))[1])
    # except wrong label data
    preds_test = model.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input)[0]
    print(len(inds_correct))
    print('The model accuracy is', len(inds_correct) / len(x_input))
    x_input = x_input[inds_correct]
    y_input = y_input[inds_correct]
    
    # use foolbox to generate 4 kinds of adversarial examples
    # fgsm, bim-a, jsma and cw
    # as the baseline uses the 4 kinds of adv to test
    # we just use the default parameters to generate adv
    # fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
    fmodel = foolbox.models.KerasModel(model, bounds=(-2.3, 2.8))
    adversarial_result = []

    training_data_folder = '../results/detector/svhn'
    if not os.path.exists(training_data_folder):
        os.makedirs(training_data_folder)

    # just generate adversarial examples by untargeted attacks
    criterion = foolbox.criteria.Misclassification()

    # we prefer to use fgsm，bim，jsma and c&w
    # but due to the high time overhead of c&w(as binary search
    # we use deepfool attack to replace it here
    # in order to reproduce our method efficiently
    adv_types_list = ['fgsm', 'bim', 'jsma', 'df']

    adversarial_result = []
    input_data = enumerate(zip(x_input, y_input))
    for adv_type in adv_types_list:
        counter = 1
        if adv_type == 'fgsm':
            attack = foolbox.attacks.FGSM(fmodel, criterion)
        elif adv_type == 'jsma':
            attack = foolbox.attacks.saliency.SaliencyMapAttack(fmodel, criterion)
        elif adv_type == 'cw':
            attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, criterion)
        elif adv_type == 'bim':
            attack = foolbox.attacks.BIM(fmodel, criterion, distance=foolbox.distances.Linfinity)
        elif adv_type == 'df':
            attack = foolbox.attacks.DeepFoolAttack(fmodel, criterion)
        elif adv_type == 'lsa':
            attack = foolbox.attacks.LocalSearchAttack(fmodel, criterion)
        elif adv_type == 'dba':
            attack = foolbox.attacks.BoundaryAttack(fmodel, criterion)
        else:
            raise Exception('Unkonwn attack method:', str(adv_type))
        
        # only need 1000 train adv samples
        # so 250 adv examples for each attack is enough
        # hard coding the bound
        while counter <= 250:
            # 使用training set
            _, (img, label) = next(input_data)
            label = np.argmax(label)

            if adv_type == 'fgsm':
                if dataset == 'mnist':
                    adversarial = attack(img, label, epsilons=1, max_epsilon=0.3)
                else:
                    adversarial = attack(img, label, epsilons=1, max_epsilon=0.03) 
                    # adversarial = attack(img, label, binary_search=True, epsilon=0.3, stepsize=0.05, iterations=10
            elif adv_type == 'bim':
                if dataset == 'mnist':
                    adversarial = attack(img, label, binary_search=False,
                                         epsilon=0.3, stepsize=0.03, iterations=10)
                else:
                    adversarial = attack(img, label, binary_search=False,
                                         epsilon=0.03, stepsize=0.003, iterations=10)
            else:
                adversarial = attack(img, label)

            if adversarial is not None:
                print('\r attack success:', counter, end="")
                adv_label = np.argmax(fmodel.predictions(adversarial))
                if adv_label != label:
                    adversarial_result.append(adversarial)
                    counter += 1

        print('\n%s attack finished.' % adv_type)
        file_name = '%s_%s_adv_examples.npy' % ("svhn", adv_type)
        np.save(training_data_folder + '/' + file_name, np.array(adversarial_result))
        adversarial_result = []


def generate_attack_cost(kmodel, dataset, attack_method):
    """ 
    Generate attack costs for benign and adversarial examples.
    """  
    
    # x_input, y_input, _, _ = get_data(dataset) # the input data min-max -> 0.0 1.0
    # (x_input, y_input), (_, _) = keras.datasets.cifar10.load_data()
    # x_input = cifar_preprocessing(x_input)
    # y_input = y_input.reshape(-1)
    
    (x_input, y_input), _ = load_svhn()
    x_input = svhn_preprocessing(x_input)
    y_input = y_input.reshape(-1)
    y_input = y_input - 1
    # except wrong label data
    preds_test = kmodel.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input)[0]
    print(len(inds_correct))
    x_input = x_input[inds_correct]
    x_input = x_input[0: 1000]
    # save_path = '../results/detector/' + dataset + '_benign'
    save_path = '../results/detector/svhn_benign'
    attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)

    # for adv_type in ['fgsm', 'bim', 'jsma', 'df']:
    #     training_data_folder = '../results/detector/svhn/'
    #     file_name = '%s_%s_adv_examples.npy' % ("svhn", adv_type)
    #     x_input = np.load(training_data_folder + file_name)
    #     save_path = '../results/detector/' + attack_method + '/' + 'svhn' + '_' + adv_type
    #     attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)


def main(args):
    dataset = args.dataset
    detector_type = args.detector 
    attack_method = args.attack

    # * load model
    # kmodel = load_model('../data/model_%s.h5' % dataset)
    # kmodel = load_model("../data/lenet5_softmax.h5")
    # kmodel = load_model("D:/Deephunter-backup-backup/deephunter/new_model/cifar10_vgg_model.194.h5")
    kmodel = get_svhn_model()
    kmodel.load_weights("../svhn_vgg16_weight.h5")
    sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    kmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    if args.init:
        print('generate training data...')
        # ! Step 1. Generate training adv
        # * use this method to generate 1000 adversarial examples from training set as training set
        # generate_training_data(kmodel, dataset)
        
        # ! Step 2. Get the attack costs of training adv
        # * use this method to obtain atatck costs on training data
        generate_attack_cost(kmodel, dataset, attack_method)
    
    # ! Step 3. Train and pred
    root_path = '../results/'
    train_benign_data_path = 'detector/' + args.attack + '/' + args.dataset + '_benign'
    # train_benign_data_path = 'detector/' + args.attack + '/' + 'svhn' + '_benign'
    adv_types_list = ['fgsm', 'bim', 'jsma', 'df']
    test_types_list = ['fgsm', 'bim-a', 'cw', 'jsma']
    train_adv_data_paths = []
    for adv_type in adv_types_list:
        train_adv_data_paths.append('detector/' + args.attack + '/' + args.dataset + '_' + adv_type)
        # train_adv_data_paths.append('detector/' + args.attack + '/' + 'svhn' + '_' + adv_type)

    if detector_type == 'knn':
        # * train k-nearest neighbors detector
        from sklearn import neighbors
        N = 100
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=N)
        print('knn based detector, k value is:', N)

        data_train, target_train = get_single_detector_train_data(root_path, train_benign_data_path, train_adv_data_paths)
        knn_model.fit(data_train, target_train)
        print('training acc:', knn_model.score(data_train, target_train))

        # * test k-nearest neighbors detector
        for test_type in test_types_list:
            # test_path = args.dataset + '_attack_iter_stats/' + args.attack + '_attack_' + test_type
            # test_path = os.path.join(root_path, test_path)
            # test_path = '../results/detector/mnist_softmax_bim_0_2'
            target_acc_list = []
            for truth_index in range(4,10):
                acc_list = []
                # truth_index = 4
                # for test_path in ['../results/detector/mnist_softmax_mmd_%s_1_20'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_21'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_22'%truth_index,
                #                   '../results/detector/mnist_softmax_mmd_%s_1_23'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_24'%truth_index,
                #                   '../results/detector/mnist_softmax_mmd_%s_1_25'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_26'%truth_index,
                #                   '../results/detector/mnist_softmax_mmd_%s_1_27'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_28'%truth_index,
                #                   '../results/detector/mnist_softmax_mmd_%s_1_29'%truth_index,'../results/detector/mnist_softmax_mmd_%s_1_30'%truth_index,
                #                   '../results/detector/mnist_softmax_mmd_%s_1_31'%truth_index]:
                target_list = np.arange(10)
                target_list = np.delete(target_list, truth_index)
                all_test_path = []
                for target in target_list:
                    all_test_path.append("../results/detector/svhn_softmax_mmd_%s_%s"%(truth_index, target))
                # for iteration in range(1,32):
                    # all_test_path.append("../results/detector/mnist_softmax_0_1_iteration_%s"%iteration)
                # all_test_path = ["../results/detector/cifar_softmax_nbc"]
                all_test_path = ["../results/detector/mnist_softmax_hda_0"]
                for idx, test_path in enumerate(all_test_path):
                # for test_path in ['../results/detector/mnist_softmax_mmd_%s_0'%truth_index,'../results/detector/mnist_softmax_mmd_%s_2'%truth_index]:
                    with open(test_path) as f:
                        lines = f.read().splitlines()
                    test_lines_list = []
                    for line in lines:
                        try:
                            test_lines_list.append(int(line))
                        except:
                            raise Exception('Invalid data type in test data:', line)
                    # assert len(test_lines_list) == 1000

                    test_lines_list = np.expand_dims(test_lines_list, axis=1)
                    if len(test_lines_list) == 0:
                        acc = 0
                    else:
                        result = knn_model.predict(test_lines_list)
                        acc = sum(result) / len(result)
                    # print(len(result))
                    # print(sum(result))
                    acc_list.append(acc)
                    if test_type is 'benign':
                        print('For knn based detector, detect acc on %s samples is %.4f'%(test_type, 1-acc))
                    else:
                        print('For knn based detector, detect acc on %s samples is %.4f'%(test_type, acc), truth_index, idx)
                target_acc_list.append(acc_list)
                # print(acc_list)   
                # np.save("./mnist_01_iteration.npy", acc_list) 
            # np.save("./target_aad_acc.npy", target_acc_list)
            break
    else:
        # * z-score based detector
        from scipy import stats
        from scipy.special import inv_boxcox
        from scipy.stats import normaltest

        data_train, _ = get_single_detector_train_data(root_path, train_benign_data_path, train_adv_data_paths)
        data_train = data_train[:1000].reshape(1, len(data_train[:1000]))[0] # only need benign data
        data_train[data_train == 0] = 0.01
        _, p_value = normaltest(data_train)
        z_score_thrs = 1.281552
        arr_mean = np.mean(data_train)
        arr_std = np.std(data_train)

        if p_value < 0.05:
            print('raw attack cost dist not pass the normal test, need boxcox')
            _, lmd = stats.boxcox(data_train)
            thrs_min = arr_mean - z_score_thrs * arr_std
            thrs_min = inv_boxcox(thrs_min, lmd)
        else:
            print('raw attack cost dist pass the normal test')
            thrs_min = arr_mean - z_score_thrs * arr_std

        # * test z-score detector
        for test_type in test_types_list:
            test_path = args.dataset + '_attack_iter_stats/' + args.attack + '_attack_' + test_type
            test_path = os.path.join(root_path, test_path)
            with open(test_path) as f:
                lines = f.read().splitlines()
            test_lines_list = []
            for line in lines:
                try:
                    test_lines_list.append(int(line))
                except:
                    raise Exception('Invalid data type in test data:', line)
            assert len(test_lines_list) == 1000

            result = [1 if _ < thrs_min else 0 for _ in test_lines_list] # for ensemble detector, need np.intersect1d and np.union1d
            acc = sum(result) / len(result)
            if test_type is 'benign':
                print('For z-score based detector, detect acc on %s samples is %.4f'%(test_type, 1-acc))
            else:
                print('For z-score based detector, detect acc on %s samples is %.4f'%(test_type, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help="Dataset to use; either 'mnist', 'cifar'",
        required=False, type=str, default='mnist',
    )
    parser.add_argument(
        '--init',
        help="If this is the first time to run this script, \
            need to generate the training data and attack costs.",
        action='store_true',
    )
    parser.add_argument(
        '-d', '--detector',
        help="Detetor type; either 'knn' or 'zscore'.",
        required=True, type=str,
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; recommanded to use JSMA, BIM or BIM2.",
        required=True, type=str,
    )
    args = parser.parse_args()
    assert args.detector in ['knn', 'zscore'], "Detector parameter error"
    assert args.attack in ['JSMA', 'BIM', 'BIM2'], "Attack parameter error"
    
    args = parser.parse_args()
    main(args)