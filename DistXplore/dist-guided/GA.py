import sys
sys.path.append('.')
import numpy as np
import copy
import random
import os
import time
from ImgMutators import Mutators as pixel_Mutators
from mutators import Mutators
from tensorflow.keras import backend as K
import cal_mmd
import tqdm
import matplotlib.pyplot as plt


def create_image_indvs_mmd(imgs, num): # num * clusters, each cluster[n*images]  (done)
    indivs = []
    shape = imgs.shape
    if len(shape) < 4:
        shape = (shape[0], shape[1], shape[2], 1)
        imgs = np.array(imgs).reshape(shape)
    
    for i in range(num):
        tmp_indivs = []
        for img in imgs:
            tmp_indivs.append(Mutators.image_random_mutate_ga(img))
        indivs.append(np.array(tmp_indivs).reshape(shape)) 
    return indivs

def create_image_indvs_mmd_pixel(imgs, num):
    indivs = []
    shape = imgs.shape
    if len(shape) < 4:
        shape = (shape[0], shape[1], shape[2], 1)
        imgs = np.array(imgs).reshape(shape)

    for i in range(num):
        tmp_indivs = []
        for img in imgs:
            tmp_indivs.append(pixel_Mutators.mutate(img, img))
        indivs.append(np.array(tmp_indivs).reshape(shape))    
    return indivs

def create_image_indvs_mmd_pixel_ori(imgs, num):
    indivs = []
    indivs.append(np.array(imgs).reshape(len(imgs), 28, 28, 1))
    for i in range(num-1):
        tmp_indivs = []
        for img in imgs:
            img = img.reshape(28, 28, 1)
            tmp_indivs.append(pixel_Mutators.mutate(img, img))
        indivs.append(np.array(tmp_indivs))    
    return indivs

def predict(input_data, model):
    inp = model.input
    layer_outputs = []
    for layer in model.layers[1:]:
        layer_outputs.append(layer.output)
    # functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    functor = K.function(inp, layer_outputs)
    outputs = functor([input_data])
    return outputs

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

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    temp /= 255.
    mean = [0.44154793, 0.44605806, 0.47180146]
    std = [0.20396256, 0.20805456, 0.20576045]
    for i in range(temp.shape[-1]):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]       
    return temp


class Population():

    def __init__(self, individuals,
                 mutation_function,
                 mutation_function_pixel,
                 fitness_compute_function,
                 save_function,
                 groud_truth,
                 subtotal,
                 first_attack,
                 seed,
                 max_iteration,
                 mode,
                 model,
                 nes,
                 pop_num,
                 target,
                 type_,
                 tour_size=20, cross_rate=0.5, mutate_rate=0.01, max_trials = 50, max_time=30):
        
        self.individuals = individuals # a list of individuals, current is numpy
        self.mutation_func = mutation_function
        self.mutation_func_pixel = mutation_function_pixel
        self.fitness_fuc = fitness_compute_function
        self.save_function = save_function
        self.ground_truth = groud_truth
        self.subtotal = subtotal
        self.firstattack = first_attack
        self.seed =seed
        self.first_iteration_used = max_iteration
        self.mode = mode
        self.model = model
        self.nes = nes
        self.pop_num = pop_num
        self.target = target
        self.type_ = type_

        
        self.tournament_size = tour_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.first_time_attacked = max_time
        
        
        self.fitness = None   # a list of fitness values
        self.pop_size = len(self.individuals)
        self.order = []
        self.best_fitness = -1000
        self.best_mmds = 99
        self.min_idx = 0
        self.record_mmds = []
        self.mutator_idx = []
        self.round = 0
        self.seed_output = ''
        self.mmds = None
        
        # other info
        self.switch_counter = 0
        self.other_mmds = [[] for i in range(len(self.nes))]
        self.preprocess_dic = {
            'cifar': cifar_preprocessing,
            'mnist': mnist_preprocessing,
            'svhn': svhn_preprocessing,
            'fmnist': mnist_preprocessing
        }
        self.preprocess = self.preprocess_dic[self.type_]
    
    def evolve_process(self, seed_output, target):
        start_time = time.time()
        i = 0
        counter = 0
        success = 0
        plot_file = open(os.path.join(seed_output, 'plot.log'), 'a+')
        plot_file.write("This is for target %d(non-target:-1,target:0-9). \n"%target)
        self.seed_output = seed_output

        self.mmds = self.cal_cur_mmd(self.individuals, self.nes[int(target)])
       
        while True:
            
            self.record_mmds.append(self.best_mmds) # best mmd to target 

            if len(self.record_mmds) > 7:
                his_best_mmd = np.mean(self.record_mmds[-7:])
                print('best - his {}'.format(abs(self.best_mmds - his_best_mmd))) #--debug--
                if abs(self.best_mmds - his_best_mmd) < 0.0005 and self.switch_counter == 0:
                    self.mode = 'pixel'
                    self.switch_counter = 1
                    print('Switching to pixel level GA')
                    # self.individuals = create_image_indvs_mmd_pixel_ori(self.individuals[np.argmin(self.mmds)], self.pop_num)
                    with open(seed_output + '/switching_point.txt', 'w') as f:
                        f.write(str(i)+'\n')

            
            if i > self.first_iteration_used:
                print("reach the max iteration")
                break
            if time.time()-start_time > self.first_time_attacked:
                print("reach the max time")
                break
            
            i += 1
            self.round = i
            results = self.evolvePopulation()
          
            cur_time = time.time()
            if results is None:
                print("Used time: {} ,Total generation: {}, best mmd: {:.3f}".format(cur_time-start_time,i, self.best_mmds))
                plot_file.write("Used time: {} ,Total generation: {}, best mmd: {:.3f} \n"
                                .format(cur_time-start_time,i, self.best_mmds))
                # print("Used time:%.4f ,Total generation: %d, best fitness:%.9f"%(cur_time-start_time,i, self.best_fitness))
                # plot_file.write("Used time:%.4f ,Total generation: %d, best fitness:%.9f \n"
                #                 %(cur_time-start_time,i, self.best_fitness))
            else:
                results = np.array(results)
                max_ = 0
                index_ = [0, 0]
                
                # print('prob {}'.format(results[:,-3]))
                for idx, cluster_prob in enumerate(results[:,-3]):
                    tmp = max(cluster_prob)
                    if tmp > max_:
                        max_ = tmp
                        index_[0] = idx # which cluster
                        index_[1] = np.argmax(cluster_prob) # which specific image
                           
                # highest_prob = max(results[-2])  # interest_probs
                # index = np.argmax(results[-2])
                highest_prob = max_
                index = index_
                # print('results len: {} (GA.py line98)'.format(len(results)))
                # print('idx 1: {}'.format(index[1]))
                pred = results[:,0][index[0]][index[1]]
                cur_mutator_num = 0
                for cluster in results[:,-2]:
                    cur_mutator_num += len(cluster)
                counter += cur_mutator_num
                print("Used time: {} , Total generation: {}, best mmd {:.3f}, find {} mutators, highest_prob is {:.3f}, prediction is {}".format(cur_time-start_time, i, self.best_mmds, cur_mutator_num, highest_prob, pred)) 
                plot_file.write("Used time: {}, Total generation: {}, best mmd {:.3f}, find {} mutators, highest_prob is {:.3f}, prediction is {} \n".format(cur_time-start_time, i, self.best_mmds, len(results[:,-2]), highest_prob, pred))
                # print("Used time:%.4f ,Total generation: %d, best fitness:%.9f, find %d mutators,highest_prob is %.5f,prediction is %d"%(cur_time-start_time,i, self.best_fitness,len(results[-1]),highest_prob,pred)) 
                # plot_file.write("Used time:%.4f ,Total generation: %d, best fitness:%.9f, find %d mutators,highest_prob is %.5f,prediction is %d \n"%(cur_time-start_time,i, self.best_fitness,len(results[-1]),highest_prob,pred))
                
                self.save_function(results, i)
                success = 1
                
                cluster_index = results[:,-1]    
                
                if self.firstattack == 1:
                    self.first_time_attacked = time.time()-start_time
                    self.first_iteration_used = i
                    break
                
                else: 
                    wrong_pred_indexes = []
                    for idx, re in enumerate(results[:,-2]):
                        wrong_pred_indexes.append(re)
                    # print('wrong_pred_indexes{}'.format(wrong_pred_indexes))
                    # if self.mode == 'pixel':
                    #     new_invs = create_image_indvs_mmd_pixel(self.seed, len(wrong_pred_indexes))
                    # else:
                    #     new_invs = create_image_indvs_mmd(self.seed, len(wrong_pred_indexes))

                    # print('new_invs len {}'.format(len(new_invs)))
                    # for j in range(len(new_invs)):
                        
                    #     # --replace entire cluster--
                    #     # self.individuals[cluster_index[j]] = new_invs[j] 
                        
                    #     # --replace single image--
                    #     for wrong_pred in wrong_pred_indexes: 
                    #         if cluster_index[j] == self.min_idx:
                    #             j+=1
                    #             # tmp_idx = wrong_pred[:len(wrong_pred)//2]
                    #             # for img_index in tmp_idx:
                    #             #     self.individuals[cluster_index[j]][img_index] = new_invs[j][img_index] 
                    #         else:
                    #             for img_index in wrong_pred:
                    #             # print('cur wrong cluster_idx: {}, img_idx: {}'.format(cluster_index[j], img_index))
                    #                 self.individuals[cluster_index[j]][img_index] = new_invs[j][img_index]               

                self.mutator_idx.append(i)
            
            if i % 100 == 0:
                plot_file.flush()
        
        whole_pth = seed_output.split('/')
        # print(whole_pth)
        chart_name = whole_pth[-1][6] + '_' + whole_pth[2]
        
        mmd_records_pth = seed_output + '/mmd_records/'
        if not os.path.exists(mmd_records_pth):
            os.makedirs(mmd_records_pth)
        
        with open(mmd_records_pth + chart_name + '_target_mmd_records.txt', 'w') as f:
            for record in self.record_mmds[1:]:
                f.write(str(record)+'\n')
        
        with open(seed_output + '/mutator_idx.txt', 'w') as f:
            for idx in self.mutator_idx:
                f.write(str(idx)+'\n')        

        for idx in range(len(self.other_mmds)):
            if idx != int(target):
                with open(mmd_records_pth + chart_name + '_' + str(idx) + '_mmd_records.txt', 'w') as f:
                    for record in self.other_mmds[idx]:
                        f.write(str(record)+'\n')

        print('Success' if success else 'Failed')
        print("Total mutators:%d "%counter)
        plot_file.write("success:%d \n"%success)
        plot_file.write("Total mutators:%d \n"%counter)
        plot_file.flush()
        plot_file.close()
        
    def crossover(self, inds1, inds2): # inds1[N, 1, 28, 28, 1]

        new_ind = []
        
        mask = np.random.uniform(0,1,len(inds1))
        mask[mask < self.cross_rate] = 0
        mask[mask >= self.cross_rate] = 1
        
        one = [int(x) for x in 1-mask]
        two = [int(x) for x in mask]
        
        for idx, i in enumerate(one):
            if i == 1:
                new_ind.append(inds1[idx])

        for idx, i in enumerate(two):
            if i == 1:
                new_ind.append(inds2[idx])

        return np.array(new_ind)

    def crossover_pixel(self, ind1, ind2):
        shape = ind1.shape
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        new_ind = np.zeros_like(ind1)
        
        mask = np.random.uniform(0,1,len(ind1))
        mask[mask < self.cross_rate] = 0
        mask[mask >= self.cross_rate] = 1
        
        new_ind = ind1 * (1-mask) + ind2 * mask

        return np.reshape(new_ind,shape)
    
    def crossover_pixel_v2(self, ind1, ind2):
        new_ind = []
        for idx, ind in enumerate(ind1):
            shape = ind.shape
            tmp_ind1 = ind.flatten()
            tmp_ind2 = ind2[idx].flatten()
            tmp_new_ind = np.zeros_like(tmp_ind1)
            
            mask = np.random.uniform(0,1,len(tmp_ind1))
            mask[mask < self.cross_rate] = 0
            mask[mask >= self.cross_rate] = 1
            
            tmp_new_ind = tmp_ind1 * (1-mask) + tmp_ind2 * mask
            new_ind.append(np.reshape(tmp_new_ind,shape))
        return np.array(new_ind)

    def cal_cur_mmd(self, indvs, nes):
        mmds = []
        for idx, ind in tqdm.tqdm(enumerate(indvs)): 
            tmp_outputs = predict(self.preprocess(ind), self.model)
            mmds.append(cal_mmd.cal_mmd(tmp_outputs[-2], nes).cpu().detach().numpy())
        return mmds
    
    def cal_best_mmd(self, ind, nes):
        tmp_outputs = predict(self.preprocess(ind), self.model)
        return cal_mmd.cal_mmd(tmp_outputs[-2], nes).cpu().detach().numpy()

    def evolvePopulation(self):

        if self.pop_size % self.subtotal == 0:
            group_num = int(self.pop_size / self.subtotal) 
        else:
            group_num = int(self.pop_size / self.subtotal) + 1
        
        # Divide initial population into several small group and start a tournament
        sorted_mmds = []
        index_ranges = []
        prepare_list = locals()
        for i in range(group_num):
            if i != group_num-1:
                prepare_list['sorted_mmds_indexes_'+str(i+1)] = sorted(range(i*self.subtotal,(i+1)*self.subtotal), 
                                                       key=lambda k: self.mmds[k], reverse=False)
                index_ranges.append((i*self.subtotal,(i+1)*self.subtotal))
            else:
                prepare_list['sorted_mmds_indexes_'+str(i+1)] = sorted(range(i*self.subtotal,self.pop_size), 
                                                       key=lambda k: self.mmds[k], reverse=False)
                index_ranges.append((i*self.subtotal,self.pop_size))
                
            sorted_mmds.append(prepare_list['sorted_mmds_indexes_'+str(i+1)])
            
        new_indvs = []
        for j in range(group_num):
            sorted_mmds_indexes = sorted_mmds[j]
            best_index = sorted_mmds_indexes[0]   # min mmds
            # print('best_index: {}'.format(best_index))
            (start,end) = index_ranges[j]
            base_index = self.subtotal * j
            for i in range(start,end):
                item = self.individuals[i]
                if i == best_index:  # keep best
                    new_indvs.append(item)
                else:
                    # self.tournament_size should be smaller than end-start
                    order_seq1 = np.sort(np.random.choice(np.arange(start-base_index,end-base_index), self.tournament_size, replace=False))
                    order_seq2 = np.sort(np.random.choice(np.arange(start-base_index,end-base_index), self.tournament_size, replace=False))
                    # pick two best candidate from this tournament

                    first_individual = self.individuals[sorted_mmds_indexes[order_seq1[0]]]
                    second_individual = self.individuals[
                        sorted_mmds_indexes[order_seq2[0] if order_seq2[0] != order_seq1[0] else order_seq2[1]]]
                    # print('first_individual {} (GA.py line235)'.format(first_individual.shape))
                    # print('second_individual {} (GA.py line235)'.format(second_individual.shape))
                    # crossover
                    if self.mode == 'pixel':
                        ind = self.crossover_pixel_v2(first_individual, second_individual)
                    else:
                        ind = self.crossover(first_individual, second_individual)
                    # print('ind shape {} (GA.py line235)'.format(ind.shape))
                    # print('ind [0] {} (GA line227)'.format(ind[0].shape))
                    if random.uniform(0, 1) < self.mutate_rate:
                        tmp_indivs = []
                        for i in ind:
                            if self.mode == 'pixel':
                                tmp_indivs.append(self.mutation_func_pixel(i)) # i.reshape(28,28,1)
                            else:    
                                tmp_indivs.append(self.mutation_func(i))    
                        ind = np.array(tmp_indivs)                 
                    
                    new_indvs.append(ind)

        self.individuals = new_indvs
        
        
        results = self.fitness_fuc(self.individuals)
        
        wrong_pred_indexes = results[-3]
        self.mmds = results[-1]
        
        self.best_mmds = min(self.mmds)
        self.min_idx = np.argmin(self.mmds) 
        # print('best_mmds: {}, self.min_idx {}'.format(self.best_mmds, self.min_idx))
        # record mmd to other label
        for idx in range(len(self.nes)):
            if idx != int(self.target):
                self.other_mmds[idx].append(self.cal_best_mmd(self.individuals[self.min_idx], self.nes[idx]))


        _result = results[:-1]
        # print('len wrong_pred_indexes: {} (GA.py line178)'.format(len(wrong_pred_indexes)))
        # print('len mmds: {} (GA.py line178)'.format(len(self.mmds)))
        # print('len _result: {} (GA.py line178)'.format(len(_result)))
        # Find desired mutator successfully
        tmp_results = []
        name = "{}_mmd{:.3f}".format(self.round, self.best_mmds)
        save_pth = self.seed_output + '/best_mmds'
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)        
        np.save(os.path.join(save_pth, name + '.npy'), self.individuals[self.min_idx])
        
        for idx, wrong_idx in enumerate(wrong_pred_indexes):
            
            # print('len(wrong_idx): {} (GA.py line172)'.format(len(wrong_idx)))
            if len(wrong_idx) > 0:
                res_list = []
                for res_idx in range(len(_result)):
                    res_list.append(_result[res_idx][idx])
                tmp_results.append(res_list)
            # print('len(tmp_results): {} (GA.py line172)'.format(len(tmp_results)))    
        if len(tmp_results) > 0:
            return tmp_results
        
        # Fail to find a mutator
        
        return None
