from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import shutil

from keras.models import load_model
import tensorflow as tf
import os

sys.path.append('../')

from keras import Input
from deephunter.coverage import Coverage
from sa import *

from keras.applications import MobileNet, VGG19, ResNet50
from keras.applications.vgg16 import preprocess_input,VGG16

import random
import time
import numpy as np
from deephunter.image_queue import ImageInputCorpus, TensorInputCorpus
from deephunter.fuzzone import build_fetch_function

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from keras.utils.generic_utils import CustomObjectScope
import codecs
from keras.datasets import mnist
from deephunter.utils.get_model import *



def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    #print("tempshape:",temp.shape)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    #temp=temp.reshape(1,28,28,1)
    temp = temp.astype('float32')
    temp /= 255
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

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    #temp=temp.reshape(1,32,32,3)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def mnist_preprocessing1(x_test):
    x_test=x_test.reshape(1,28,28,1)
    return x_test

def cifar_preprocessing1(x_test):
    x_test=x_test.reshape(1,32,32,3)
    return x_test

def imgnt_preprocessing(x_test):
    return x_test

model_weight_path = {
    'vgg16': "./new_model/cifar10_vgg_model.194.h5",
    'resnet20': "/data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_prob_kmnc2.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5",
    'allconv': "./new_model/lenet5_softmax.h5",
    # 'allconv':"../../deephunter_data/pretrained_models/mnist_lenet_5/mnist_lenet5_9.h5",
    'lenet5_opt': "./new_model/lenet5_for_opt.h5",
    'fashion_lenet5':"./new_model/fm_lenet5.h5",
    'svhn_net':"./utils/svhn_vgg16_weight.h5",
    'bug_insert':"../../dnn_bug_insert/bug_insert_models/layer_0_bug_insert_models/model_24.h5",
    'ws_model':"../../deepmutationoperators-master/DeepMutationOperators-master/insert_bug_model/lenet_WS_model.h5",
    'mnist_lenet4':"-",
    'fmnist_lenet4':"-",
    'cifar_resnet':"-",
    'svhn_vgg':"-",
    'svhn_resnet':"-"
}

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_90000.pickle",
    'resnet20': "/data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_prob_kmnc2.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/0_600000.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'vgg19': "./profile/imagenet/profiling/vgg19_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle",
    'allconv': "./new_model_profile/lenet5_softmax.pickle",
    # 'allconv':"./mnist_9.pickle",
    'lenet5_opt': "./new_model_profile/lenet5_for_opt.pickle",
    'fashion_lenet5':"./new_model_profile/fm_lenet5.pickle",
    'svhn_net':"./new_model_profile/svhn.pickle",
    'bug_insert':"./new_model_profile/mnist_lenet_layer_0_insert_bug_model_24.pickle",
    'ws_model':"./ws_model_profile.pickle",
    'mnist_lenet4':"/data/wlt/deephunter/deephunter/new_model_profile/mnist_lenet4.pickle",
    'fmnist_lenet4':"/data/wlt/deephunter/deephunter/new_model_profile/fmnist_lenet4.pickle",
    'cifar_resnet':"/data/wlt/deephunter/deephunter/new_model_profile/cifar_resnet.pickle",
    'svhn_vgg':"/data/wlt/deephunter/deephunter/new_model_profile/svhn.pickle",
    'svhn_resnet':"/data/wlt/deephunter/deephunter/new_model_profile/svhn_resnet.pickle"
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'vgg19': imagenet_preprocessing,
    'resnet50': imgnt_preprocessing,
    'allconv': mnist_preprocessing,
    'lenet5_opt': mnist_preprocessing,
    'fashion_lenet5': mnist_preprocessing,
    'svhn_net': svhn_preprocessing,
    'bug_insert': mnist_preprocessing,
    'ws_model': mnist_preprocessing,
    'mnist_lenet4':mnist_preprocessing,
    'fmnist_lenet4':mnist_preprocessing,
    'cifar_resnet':cifar_preprocessing,
    'svhn_vgg':svhn_preprocessing,
    'svhn_resnet':svhn_preprocessing
}

shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (28, 28, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (256, 256, 3),
    'allconv': (28, 28, 1), 
    'lenet5_opt': (28, 28, 1),
    'fashion_lenet5':(28, 28, 1),
    'svhn_net':(32,32,3),
    'bug_insert':(28, 28, 1),
    'ws_model':(28, 28, 1),
    'mnist_lenet4':(28, 28, 1),
    'fmnist_lenet4':(28, 28, 1),
    'cifar_resnet':(32,32,3),
    'svhn_vgg':(32,32,3),
    'svhn_resnet':(32,32,3)
}
metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10,
    'lsa': 10
}
execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'],
    'allconv': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5_opt': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'fashion_lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'svhn_net': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'bug_insert': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'ws_model' : ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mnist_lenet4':['input', 'flatten', 'activation', 'batch', 'dropout'],
    'fmnist_lenet4':['input', 'flatten', 'activation', 'batch', 'dropout'],
    'cifar_resnet':['input', 'flatten', 'activation', 'batch', 'dropout'],
    'svhn_vgg':['input', 'flatten', 'activation', 'batch', 'dropout'],
    'svhn_resnet':['input', 'flatten', 'activation', 'batch', 'dropout']
}

def metadata_function(meta_batches):
    return meta_batches

def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.image_random_mutate(seed, batch_num)

    return func

def objective_function(seed, names):
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    assert (names is not None)
    results = []
    if len(metadata) == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
            results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                results.append(names[count] + adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results

def iterate_function(names,model,args):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches,label_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):
            # print(mutated_coverage_list.shape)
            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, l0_batches[idx], linf_batches[idx])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx],label_batches[idx], True, suffix=i)
                    #queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
            else:

                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, label_batches[idx],False)
                successed = successed or result
        return bug_found, successed

    return func

def dry_run(indir, fetch_function, coverage_function, queue,model,args):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    for idx,seed_name in enumerate(seed_lis):
        tf.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.array([np.load(path)[0]])
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        #input_batches = img[1:2]
        print("test",img.shape)
        preprocess1 = preprocess_dic[args.model]
        # print(img.shape)
        input_batches=img
        #print(input_batches.shape)
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches, original_imgs = fetch_function((0, input_batches, 0, 0, 0,0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches,model,args, original_imgs)
        #print(coverage_list)
        metadata_list = metadata_function(metadata_batches)
        # Create a new seed
        # print("ttt", coverage_list)
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0])
        # print(input.coverage)
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img,np.array([idx]), False, True, seed_name)


if __name__ == '__main__':
    
    global_ptr = np.zeros(1000)

    start_time = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory')
    parser.add_argument('-o', help='output directory')

    parser.add_argument('-model', help="target model to fuzz", choices=['allconv','vgg16', 'resnet20', 'mobilenet', 'vgg19','resnet20new','cifarcnn','lenet5_fmnew','lenet5new','svhnresnet',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5','fashion_lenet5','svhnnet','svhnnew','lenet4fm','mnist_lenet4', 'fmnist_lenet4','cifar_resnet','svhn_vgg','svhn_resnet'], default='lenet5')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann','lsa'], default='lsa')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=10000000)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type=int)
    # parser.add_argument('-ann_threshold', help="Distance below which we consider something new coverage.", type=float,
    #                     default=1.0)
    parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=0)
    parser.add_argument('-select', help="test selection strategy",
                        choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("-gpu_index")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )

    
    args = parser.parse_args()
    os.environ["VISIBLE_CUDA_DEVICES"] = args.gpu_index
    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))
    os.makedirs(os.path.join(args.o, 'queue_seeds'))
    os.makedirs(os.path.join(args.o, 'crash_seeds'))
    os.makedirs(os.path.join(args.o, 'crash_labels'))
    os.makedirs(os.path.join(args.o, 'queue_labels'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    elif args.model == 'vgg19':
        model = VGG19(input_tensor=input_tensor)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    elif args.model == 'mnist_lenet4':
        model = mnist_lenet4()
    elif args.model == 'fmnist_lenet4':
        model = fmnist_lenet4()
    elif args.model == 'cifar_resnet':
        model = cifar_resnet20()
        print(model.summary())
    elif args.model == 'svhn_vgg':
        model = svhn_vgg16()
    elif args.model == 'svhn_resnet':
        model = svhn_resnet20()
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    #profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'))
    profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'), encoding='iso-8859-1')

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)
    
    # (x_train,y_train),(x_test,y_test) = mnist.load_data()
    # layer_names = ["dense_2"]
    # seed_list = []
    # for i in range(len(os.listdir(args.i))):
    #     seed_list.append(np.load(os.path.join(args.i, os.listdir(args.i)[i])))
    # train_lsa = fetch_lsa(model, x_train, args.i, "test", layer_names,args)
    # train_lower=np.amin(train_lsa)
    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=profile_dict, exclude_layer=exclude_layer_list)
    #print(type(coverage_handler))

    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    if args.quantize_test == 1:
        model_names = os.listdir(args.quan_model_dir)
        model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
        if args.model == 'mobilenet':
            import keras

            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                models = [load_model(m) for m in model_paths]
        else:
            models = [load_model(m) for m in model_paths]
        fetch_function = build_fetch_function(coverage_handler, preprocess, models)
        model_names.insert(0, args.model)
    else:
        fetch_function = build_fetch_function(coverage_handler, preprocess)
        model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage
    #print(type(coverage_function))
    # The function to perform the mutation from one seed
    mutation_function = image_mutation_function(args.batch_num)

    # The seed queue
    if args.criteria == 'fann':
        queue = TensorInputCorpus(args.o, args.random, args.select, cri, "kdtree")
    else:
        queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    # Perform the dry_run process from the initial seeds
    dry_run(args.i, dry_run_fetch, coverage_function, queue,model,args)

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    image_iterate_function = iterate_function(model_names,model,args)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function, fetch_function,
                    image_iterate_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration,model,args, global_ptr)

    shutil.rmtree(os.path.join(args.o, 'queue'))
    shutil.rmtree(os.path.join(args.o, 'queue_labels'))

    print('finish', time.time() - start_time)
    
    with codecs.open('/data/zyh/deephunter/deephunter/result.txt', mode='a', encoding='utf-8') as file_txt:
            file_txt.write('\n'+str(args.o)+'\t'+str(time.time() - start_time))

