from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import collections
from keras import backend as K
from collections import OrderedDict
import sys
from sa import *
from keras.datasets import mnist, cifar10, fashion_mnist
from scipy import io

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    #temp=temp.reshape(1,32,32,3)
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

    x_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['X'] # 73257
    y_train = io.loadmat('/data/wlt/svhn/train_32x32.mat')['y']

    x_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['X'] # 26032 
    y_test = io.loadmat('/data/wlt/svhn/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)

class Coverage():

    def __init__(self, model, criteria, k = 10, profiling_dict={}, exclude_layer=['input', 'flatten']):

        self.model = model
        if criteria == 'nbc':
            self.k = k + 1
            self.bytearray_len = self.k * 2
        elif criteria == 'snac':
            self.k = k + 1
            self.bytearray_len = self.k
        elif criteria == 'nc':
            self.k =k
            self.bytearray_len = 1
        else:
            self.k = k
            self.bytearray_len = self.k

        self.criteria = criteria
        self.profiling_dict = profiling_dict


        self.layer_to_compute = []
        self.outputs = []
        self.layer_neuron_num = []
        self.layer_start_index = []
        self.start = 0

        num = 0
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.layer_start_index.append(num)
                self.layer_to_compute.append(layer.name)
                self.outputs.append(layer.output)
                self.layer_neuron_num.append(layer.output.shape[-1])
                num += int(layer.output.shape[-1] * self.bytearray_len)
                #print(layer.name,int(layer.output.shape[-1] * self.bytearray_len))
        self.outputs.append(self.model.layers[-1].output)

        self.total_size = num

        self.cov_dict = collections.OrderedDict()

        inp = self.model.input
        self.functor = K.function([inp] + [K.learning_phase()], self.outputs)



    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def predict(self, input_data):
        outputs = self.functor([input_data, 0])
        return outputs

    def lsa_update_coverage(self, outputs, ptr, model, args, ori_imgs):
        if args.model == 'vgg16' or args.model == 'cifar_resnet':
            (x_train,y_train),(x_test,y_test) = cifar10.load_data()
            x_train = cifar_preprocessing(x_train)
        elif args.model == 'svhn_vgg' or args.model == 'svhn_resnet':
            (x_train,y_train),(x_test,y_test) = load_svhn()
            x_train = svhn_preprocessing(x_train)
        elif args.model == 'mnist_lenet4' or args.model == 'allconv':
            (x_train,y_train),(x_test,y_test) = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_train = (x_train / 255.0)
        elif args.model == 'fmnist_lenet4' or args.model == "fashion_lenet5":
            (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_train = (x_train / 255.0)
        # x_train = cifar_preprocessing(x_train)
        layer_names = ["dense_2"]
        if args.model == 'cifar_resnet' or args.model == 'svhn_resnet':
            layer_names =["dense_1"]
        seed_list = []
        # for i in range(len(os.listdir(args.i))):
            # seed_list.append(np.load(os.path.join(args.i, os.listdir(args.i)[i]))[0])
        # seed_list = np.array(seed_list)
        # print(seed_list)
        # print("test", ori_imgs.shape)
        # print(model.summary())
        sa = fetch_lsa(model, x_train, ori_imgs, "test", layer_names,args)
        # print(sa)
        get_sc(0, 2000, 1000, sa, ptr)
        # print("2", ptr.shape)

    def kmnc_update_coverage(self, outputs, ptr):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

                    mean_value = profiling_data_list[0]
                    std = profiling_data_list[2]
                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    unit_range = (upper_bound - lower_bound) / self.k

                    output = np.mean(layer_output[..., neuron_idx])

                    # the special case, that a neuron output profiling is a fixed value
                    # TODO: current solution see whether test data cover the specific value
                    # if it covers the value, then it covers the entire range by setting to all 1s
                    if unit_range == 0:
                        continue
                    # we ignore output cases, where output goes out of profiled ranges,
                    # this could be the surprised/exceptional case, and we leave it to
                    # neuron boundary coverage criteria
                    if output > upper_bound or output < lower_bound:
                        continue

                    subrange_index = int((output - lower_bound) / unit_range)

                    if subrange_index == self.k:
                        subrange_index -= 1

                    # print "subranges: ", subrange_index

                    id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + subrange_index
                    num = ptr[seed_id][id]
                    assert(num==0)
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num
        

    def bknc_update_coverage(self, outputs, ptr, rev):
        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]
            for seed_id, layer_output in enumerate(layer_outputs):
                # print(layer_output.shape)

                layer_output_dict = {}
                for neuron_idx in range(layer_output.shape[-1]):
                    output = np.mean(layer_output[..., neuron_idx])

                    layer_output_dict[neuron_idx] = output

                # sort the dict entry order by values
                sorted_index_output_dict = OrderedDict(
                    sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=rev))

                # for list if the top_k > current layer neuron number,
                # the whole list would be used, not out of bound
                top_k_node_index_list = sorted_index_output_dict.keys()[:self.k]

                for top_sec, top_idx in enumerate(top_k_node_index_list):
                    id = self.start + self.layer_start_index[idx] + top_idx * self.bytearray_len + top_sec
                    num = ptr[seed_id][id]
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num

    def nbc_update_coverage(self, outputs, ptr):
        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]
            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):

                    output = np.mean(layer_output[..., neuron_idx])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

                    # mean_value = profiling_data_list[0]
                    # std = profiling_data_list[2]
                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    # this version uses k multi_section as unit range, instead of sigma
                    # TODO: need to handle special case, std=0
                    # TODO: this might be moved to args later
                    k_multisection = 1000
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hypo active case, the store targets from low to -infi
                    if output < lower_bound:
                        # float here
                        target_idx = (lower_bound - output) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1


                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(target_idx)


                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(
                                target_idx)
                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue
    def snac_update_coverage(self, outputs, ptr):
        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]
            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):
                    output = np.mean(layer_output[..., neuron_idx])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

                    # mean_value = profiling_data_list[0]
                    # std = profiling_data_list[2]
                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    # this version uses k multi_section as unit range, instead of sigma
                    # TODO: need to handle special case, std=0
                    # TODO: this might be moved to args later
                    # this supposes that the unit range of boundary range is the same as k multi-1000
                    k_multisection = 1000
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(
                                target_idx)

                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num

                        continue

    def nc_update_coverage(self, outputs, ptr):
        '''
                Given the input, update the neuron covered in the model by this input.
                    This includes mark the neurons covered by this input as "covered"
                :param input_data: the input image
                :return: the neurons that can be covered by the input
                '''


        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]
            #print(layer_name,outputs[idx].shape)
            for seed_id, layer_output in enumerate(layer_outputs):
                scaled = self.scale(layer_output)
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.k:
                        id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + 0
                        #print(np.mean(scaled[..., neuron_idx]))
                        ptr[seed_id][id] = 1
    def update_coverage(self, outputs, model, args, ori_imgs):
        '''
        We implement the following metrics:
        NC from DeepXplore and DeepTest
        KMNC, BKNC, TKNC, NBC, SNAC from DeepGauge2.0.

        :param outputs: The outputs of internal layers for a batch of mutants
        :return: ptr is the array that record the coverage information
        '''
        batch_num = len(outputs[0])
        #print(outputs[0])
        #print(batch_num)
        #print(outputs[0].shape)

        ptr = np.tile(np.zeros(self.total_size, dtype=np.uint8), (batch_num,1))
        if args.criteria == "lsa":
            ptr = np.zeros((batch_num, 1001))

        if len(outputs) > 0 and len(outputs[0]) > 0:

            if self.criteria == 'kmnc':
                self.kmnc_update_coverage(outputs, ptr)
            elif self.criteria == 'lsa':
                self.lsa_update_coverage(outputs, ptr,model,args, ori_imgs)
            elif self.criteria == 'bknc':
                self.bknc_update_coverage(outputs,ptr,False)
            elif self.criteria == 'tknc':
                self.bknc_update_coverage(outputs,ptr,True)
            elif self.criteria == 'nbc':
                self.nbc_update_coverage(outputs,ptr)
            elif self.criteria == 'snac':
                self.snac_update_coverage(outputs,ptr)
            elif self.criteria == 'nc':
                self.nc_update_coverage(outputs,ptr)
            elif self.criteria == 'fann':
                # Assume the penultimate layer is the logits layer
                return np.reshape(outputs[-2], (outputs[-2].shape[0], outputs[-2].shape[-1]))
            else:
                print("* please select the correct coverage criteria as feedback:")
                print("['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann]")
                sys.exit(0)


        #print(len(ptr[0]))
        # print("testt", ptr.shape)
        return ptr



if __name__ == '__main__':
    print("main Test.")
