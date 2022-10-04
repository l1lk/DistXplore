import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from cgi import print_directory
import sys

# from pyrsistent import v
sys.path.append('.')
from GA import Population
# from GA_multi import Population_multi
# from GeneticAlgorithm2 import Population
import tensorflow as tf
from PIL import Image
# from tensorflow.keras.applications.vgg16 import preprocess_input
import random
import time
import argparse
import numpy as np

import shutil

import ntpath

from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
# from tensorflow.keras.utils.generic_utils import CustomObjectScope
# from tensorflow.keras.engine.topology import get_source_inputs

import tensorflow.keras
from mutators import Mutators
# from mutators_noise import Mutators
from ImgMutators import Mutators as pixel_Mutators
import cal_mmd

# utility
from pathlib import Path
import glob
import tqdm
# from ImgMutators import Mutators

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]       
    return temp

def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)  # final input shape = (1,224,224,3)
    return qq

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


shape_dic = {
    'vgg16': (32,32,3),
    'resnet20': (32,32,3),
    'lenet1': (28,28,1),
    'lenet4': (28,28,1),
    'lenet5': (28,28,1),
    'mobilenet': (224, 224,  3),
    'vgg19': (224, 224,  3),
    'resnet50': (224, 224,  3)
}
preprocess_dic = {
    'cifar': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'imagenet': imagenet_preprocessing,
    'svhn': svhn_preprocessing,
    'fmnist': mnist_preprocessing
}

# def image_mutation_function_mmd():
#     # Given a seed, randomly generate a batch of mutants
#     def func(seed):
#         return Mutators.image_random_mutate_ga(seed)
#     return func


# def create_image_indvs_mmd(imgs, num): # num * clusters, each cluster[n*images]  (done)
#     indivs = []
#     indivs.append(np.array(imgs).reshape(len(imgs), 1, 28, 28, 1))
#     # print('length img {}'.format(len(imgs)[0]))
#     for i in range(num-1):
#         tmp_indivs = []
#         for img in imgs:
#             tmp_indivs.append(Mutators.image_random_mutate_ga(img.reshape(28, 28, 1)))
#         indivs.append(np.array(tmp_indivs))
#     return indivs
#     # return np.array(indivs)

# def create_image_indvs_mmd_pixel(imgs, num):
#     indivs = []
#     indivs.append(np.array(imgs).reshape(len(imgs), 28, 28, 1))
#     for i in range(num-1):
#         tmp_indivs = []
#         for img in imgs:
#             img = img.reshape(28, 28, 1)
#             tmp_indivs.append(pixel_Mutators.mutate(img, img))
#         indivs.append(np.array(tmp_indivs))    
#     return indivs


def create_image_indvs_mmd(imgs, num): 
    indivs = []
    shape = imgs.shape
    print(imgs.shape)
    if shape[-1] == 784:
        imgs = imgs.reshape(-1, 28, 28)
        shape = imgs.shape
    if len(shape) < 4:
        shape = (shape[0], shape[1], shape[2], 1)
        imgs = np.array(imgs).reshape(shape)
        
    indivs.append(imgs)
    
    # print('imgs shape {}'.format(imgs.shape))
    for i in range(num-1):
        tmp_indivs = []
        for img in imgs:
            tmp_indivs.append(Mutators.image_random_mutate_ga(img))
        indivs.append(np.array(tmp_indivs).reshape(shape))
    return indivs


def predict_mmd(input_data, model):
    inp = model.input
    layer_outputs = []
    for layer in model.layers[1:]:
        layer_outputs.append(layer.output)
    functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    outputs = []
    for i in input_data:
        outputs.append(functor([input_data, 0]))
    return outputs

def build_mutate_func_mmd():
    def func(img):
        shape = img.shape
        return Mutators.image_random_mutate_ga(img).reshape(shape)
    return func

def build_mutate_func_mmd_pixel():
    def func(img):
        shape = img.shape
        return pixel_Mutators.mutate(img, img).reshape(shape)
    return func

def build_save_func_mmd(npy_output, img_output, ground_truth, seedname, target): # crash_dir, img_dir, ground_truth, seed_name, target
    def save_func(indvs, round):
        prediction = indvs[:,0]
        data = indvs[:,1]
        probs = indvs[:,2]
        indexes = indvs[:,-1]
        # for ind_idx in range(len(indvs)):
        # #     prediction.append(indvs[ind_idx][0])
        # #     data.append(indvs[ind_idx][1])
        # #     probs.append(indvs[ind_idx][2])
        #     indexes.append(indvs[ind_idx][-1])
        

        for i, item in enumerate(prediction):
            # cluster_name = "{}_{}_tar{}_gt{}_idx{}".format(round, seedname,
            #                                     target, ground_truth, indexes[i])
            # np.save(os.path.join(npy_output, cluster_name + '.npy'), data[i])
            
            #save image
            for img_idx in range(len(item)):
                name = "{}_{}_tar{}_gt{}_pred{}_{:.3f}".format(round, seedname,
                                                target, ground_truth,
                                                prediction[i][img_idx], probs[i][img_idx])
                
                np.save(os.path.join(npy_output, name + '.npy'), data[i])
        
                x = np.uint8(data[i][img_idx])
                shape = x.shape
                if shape[-1] == 1:
                    x = np.reshape(x, (shape[0], shape[1])) # shape[0], shape[1]
                img0 = Image.fromarray(x)
                img0.save(os.path.join(img_output, name + '.png'))
                # img0.save(os.path.join(img_output, name + '_image_' + str(i) + '_' + str(img_idx) + '.png'))

    return  save_func


def diff_object_func_v2(model, preprocess, label, nes, target_num = -1, threshold = 0.6, logits = True): # model, preprocess, ground_truth, target, threshold=ratio
    def func(indvs):

        array = []
    
        prediction = []
        wrong_pred_index = []
        
        evaluations = []
        prob_results = []

        out_predictions = []
        out_indvs = []
        mmds = []

        cluster_indexes = []

       
        for idx, ind in tqdm.tqdm(enumerate(indvs)): # for each ind cluster
            cluster_indexes.append(0)
        
            tmp_outputs = predict(preprocess(ind), model)
            # print('tmp_outputs length {}'.format(len(tmp_outputs[-1])))
            tmp_prediction = np.argmax(tmp_outputs[-1], axis=1)
            # print('tmp_prediction length {}'.format(tmp_prediction))
            tmp_wrong_pred_index = np.nonzero(tmp_prediction != label)[0]
        
            if logits:
                tmp_evaluations = tmp_outputs[-2]
            else:
                tmp_evaluations = tmp_outputs[-1]
            tmp_prob_results = tmp_outputs[-1]
            
            array.append(ind)
            wrong_pred_index.append(tmp_wrong_pred_index)
            evaluations.append(tmp_evaluations)
            prediction.append(tmp_prediction)
            prob_results.append(tmp_prob_results)
           
            # compute mmd
            mmds.append(cal_mmd.cal_mmd(evaluations[idx], nes).cpu().detach().numpy())

        interest_indexes = []
        interest_probs = []
        
        for idx, tmp_idx in enumerate(wrong_pred_index):
            tmp_interest_indexes = []
            tmp_interest_probs = []
            
            for i_idx in tmp_idx: # for some wrong prediction seeds(images) index in one cluster
                interest_prob = prob_results[idx][i_idx][prediction[idx][i_idx]] # For each image: [prob_0, prob_1, prob_2]
                # If it is a targeted configuration, we only care about our target goal.
                if target_num != -1 and prediction[idx][i_idx] != target_num:
                    continue
                
                
                if interest_prob > threshold:
                    
                    tmp_interest_indexes.append(i_idx)
                    tmp_interest_probs.append(interest_prob)
                    cluster_indexes[idx] = idx
            
            interest_indexes.append(tmp_interest_indexes)
            interest_probs.append(tmp_interest_probs)

            
        for idx in range(len(indvs)):
            out_predictions.append(prediction[idx][interest_indexes[idx]])
            out_indvs.append(array[idx][interest_indexes[idx]])
        
        return out_predictions, out_indvs, interest_probs, interest_indexes, cluster_indexes, mmds 

        # prediction[interest_indexes], array[interest_indexes], interest_probs, interest_indexes, fitness
    return func


'''
Original attack 
'''
def predict(input_data, model):
    inp = model.input
    layer_outputs = []
    for layer in model.layers[1:]:
        layer_outputs.append(layer.output)
    # functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    functor = K.function(inp, layer_outputs)
    outputs = functor([input_data])
    return outputs






if __name__ == '__main__':

    #=============================Initializing================================
    parser = argparse.ArgumentParser(description='coverage guided fuzzing')
    parser.add_argument('-i', help='image path')
    parser.add_argument('-o', help='seed output')
    parser.add_argument('-pop_num', help='seed output', type =int, default=1000)
    parser.add_argument('-type', help="target model fuzz", choices=['mnist','imagenet','cifar','svhn', 'fmnist'],
                        default='mnist')
    parser.add_argument('-model_type', help='Out path',
                        choices=['lenet1', 'lenet5', 'resnet20', 'mobilenet', 'vgg16', 'resnet50'], default='lenet5')

    parser.add_argument('-model', help="fuzzer for quantize")
    parser.add_argument('-ratio', type=float,help="fuzzer for quantize", default=0)
    parser.add_argument('-subtotal', type=int, default=400)

    parser.add_argument('-timeout', help="threshold for determining neuron activated", type=int, default=9999)
    parser.add_argument('-max_iteration', help="threshold for determining neuron activated", type=int, default=1000)
    parser.add_argument('-first_attack', choices=[0,1], type=int, default=0)
    parser.add_argument('-target', choices=[-1,0,1,2,3,4,5,6,7,8,9], type=int, default=-1)

    parser.add_argument('-mode', help='pixel level or image level mutation', choices=['pixel', 'image'], default='image')
    parser.add_argument('-logits', help='using logits or other layer', default=True)
    parser.add_argument('-pattern', help='cluster pattern', choices=['multi', 'single'], default='single')
    parser.add_argument('-num', help='cluster numbers', type =int, default=5)

    args = parser.parse_args()

    
    input_file =  args.i #"../seeds/mnist_seeds/0_0.npy" "dev_seeds_testing/class_0_0_seed.npy"
    seed = ntpath.basename(input_file)
    seed_name = seed.split(".")[0]

    pop_num =  args.pop_num         #1000
    type_ =  args.type            #"mnist"
    model_type = args.model_type      # "lenet5"
    model = args.model             #"../models/lenet5.h5"

    ratio = args.ratio             #0

    subtotal = args.subtotal          #400
    timeout = args.timeout           #30
    max_iteration = args.max_iteration   #1000
    first_attack =args.first_attack     #0
    target = args.target            #-1
    
    seed_output = "{0}/{1}_output_{2}".format(args.o, seed_name, target)
    
    nes = None
    
    #=============================Main Logic================================

    # Build directory structure
    if os.path.exists(seed_output):
        shutil.rmtree(seed_output)

    img_dir =  os.path.join(seed_output, 'imgs')
    crash_dir = os.path.join(seed_output, 'crashes')
    os.makedirs(crash_dir)
    os.makedirs(img_dir)
    

    # Load a model
    preprocess = preprocess_dic[type_]
    model = load_model(model)
    # if type_ == 'mobilenet' or type_ == 'resnet50':
    #     with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
    #                             'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    #         model = load_model(model)
    # else:
    #     model = load_model(model)
    
    
    # Load initial seed and give a prediction
    data = np.load(input_file)
    orig_imgs = data # for normal shape
    # orig_imgs = data.reshape(data.shape[0], 28, 28) # for mnist_v1 seeds stored with shape (100,784)
    
    # for idx in range(len(data)):
    #     orig_imgs.append(data[idx])
    #     ground_truth = np.argmax(model.predict(preprocess(data[idx:idx+1])), axis=1)[0]
    
    ground_truth = np.argmax(model.predict(preprocess(orig_imgs))[0])
    # print('ground_truth {}'.format(ground_truth))
    
    # Load nes
    
    if args.target != -1:
        if args.pattern == 'single':
            nes_path = 'single_cluster_seeds/'+ args.type + '/training_100'
        else:
            nes_path = 'multi_cluster_seeds/'+ args.type + '/training_100'
        nes_pth_list = os.listdir(nes_path)
        nes_pth_list.sort()
        # target_path = 'single_cluster_seeds/cifar10/training_100/class_'+ str(target) +'_seed.npy'
        
        
    if nes_pth_list is not None:
        nes = []
        for p in nes_pth_list:
            print('ne path {}'.format(p))
            path = os.path.join(nes_path, p)
            tmp_nes = np.load(path)
            # tmp_nes = tmp_nes.reshape(tmp_nes.shape[0], 28, 28)
            # print('tmp_nes {} (AttackSet.py line356)'.format(tmp_nes.shape)) #--debug--    
            nes_outputs = predict(preprocess(tmp_nes), model)
            nes.append(nes_outputs[-2])
    
    # print('nes {} (AttackSet.py line349)'.format(nes[0].shape)) #--debug--
 

    # check validity of target
    # if target == ground_truth:
    #     print("Target should be different from the ground truth")
    #     shutil.rmtree(seed_output)
    #     sys.exit(0)
    
    
        
    
    # Generate a batch individuals
    if args.pattern == 'single':
        inds = create_image_indvs_mmd(orig_imgs, pop_num)
    else:
        inds = []
        for i in range(args.num):
            inds.append(create_image_indvs_mmd(orig_imgs, pop_num))
    # print(inds[1].shape)
    
    # build a mutate function
    mutation_function = build_mutate_func_mmd()
    mutation_function_pixel = build_mutate_func_mmd_pixel()
    if type_ == 'svhn':
        if target == 0:
            transed_target = 9
        else:
            transed_target = target-1
        save_function = build_save_func_mmd(crash_dir, img_dir, ground_truth, seed_name, target)
        fitness_compute_function = diff_object_func_v2(model, preprocess, ground_truth, nes[target], transed_target, threshold=ratio, logits=True)
        pop = Population(inds,mutation_function,mutation_function_pixel,fitness_compute_function,save_function,ground_truth,
                     first_attack=first_attack,
                     subtotal=subtotal, max_time=timeout, seed=orig_imgs, max_iteration=max_iteration, mode=args.mode, model=model, nes=nes, pop_num=pop_num, target=target, type_=type_)
        pop.evolve_process(seed_output, target)   
    else:
        save_function = build_save_func_mmd(crash_dir, img_dir, ground_truth, seed_name, target)
        fitness_compute_function = diff_object_func_v2(model, preprocess, ground_truth, nes[target], target, threshold=ratio, logits=True)
        pop = Population(inds,mutation_function,mutation_function_pixel,fitness_compute_function,save_function,ground_truth,
                     first_attack=first_attack,
                     subtotal=subtotal, max_time=timeout, seed=orig_imgs, max_iteration=max_iteration, mode=args.mode, model=model, nes=nes, pop_num=pop_num, target=target, type_=type_)
        pop.evolve_process(seed_output, target)
   
    

# python AttackSet.py   -i dev_seeds_training/class_0_0_seed.npy  -o GA_output  -type mnist  -pop_num 1000  -subtotal 400  -model_type lenet5  -model profile/mnist/models/lenet5.h5  -timeout 50  -target 1  -first_attack 0
# 
# python AttackSet.py   -i multi_cluster_seeds/mnist/class_0_0_seed.npy  -o GA_test/GA_100_logits_mnist_multi -pop_num 100  -subtotal 50  -type mnist -model profile/lenet5_softmax.h5  -target 1  -max_iteration 30 
# python AttackSet.py   -i single_cluster_seeds/mnist/training_100/class_0_seed.npy  -o GA_test/GA_100_logits_mnist_multi -pop_num 100  -subtotal 50  -type mnist -model profile/lenet5_mnist.h5  -target 1  -max_iteration 30
# python AttackSet.py   -i single_cluster_seeds/mnist/training_100/class_0_seed.npy  -o GA_output/GA_100_logits_mnist_v2/100_50 -pop_num 100  -subtotal 50  -type mnist -model profile/lenet5_mnist.h5  -target 1  -max_iteration 30
# python AttackSet.py   -i single_cluster_seeds/cifar10/training_100/class_0_seed.npy  -o GA_output/GA_100_logits_cifar10/100_50  -pop_num 100  -subtotal 50  -type cifar10 -model profile/vgg16_cifar10.h5  -target 1  -max_iteration 30
# python AttackSet.py   -i single_cluster_seeds/svhn/training_100/class_0_seed.npy  -o GA_output/GA_100_logits_svhn/100_50  -pop_num 100  -subtotal 50  -type svhn -model profile/vgg16_svhn.h5  -target 1  -max_iteration 30



