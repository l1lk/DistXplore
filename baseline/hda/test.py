import utils
import torch
import os
import torch.nn as nn
from model.inception import InceptionV3
from GaussianKDE import GaussianKDE
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import numpy as np
from utils import get_nearest_oppo_dist, cal_gradient, fitness_score, mutation, cal_robust, cal_gradient_keras, extract_data_features
import numpy as np
from utils import calculate_fid
import time
# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras import regularizers
from keras.optimizers import SGD
import argparse
from tqdm import tqdm
from collections import Counter
from get_model import svhn_vgg16, mnist_lenet4

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



def extract_data_features(model, data, layer_index):
    """
    得到指定中间层输出
    layer_index: 指定输出层
    return features:指定层输出
    """
    sub_model = keras.models.Model(inputs=model.input, 
                                   outputs=model.get_layer(index=layer_index).output)
    features = sub_model.predict(data)
    return features

def two_step_ga(model, x_seed, y_seed, eps, local_op, n_particles = 300, n_mate = 20, max_itr = 50, alpha = 1.00):
    adv_images = x_seed.repeat(n_particles,1,1,1)
    # print("test0", np.unique(adv_images.numpy()))
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.01)
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0.0, max=1.0).detach()
    # print("test", np.unique(adv_images))
    # adv_images = adv_images*255
    # print("test", np.unique(adv_images.numpy()))
    for i in range(max_itr):
        obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op,alpha)
        sorted, indices = torch.sort(obj, dim=-1, descending=True)
        parents = adv_images[indices[:n_mate]]
        # print("test2", np.unique(parents))
        obj_parents = sorted[:n_mate]

        # Generating next generation using crossover
        m = Categorical(logits=obj_parents)
        parents_list = m.sample(torch.Size([2*n_particles]))
        parents1 = parents[parents_list[:n_particles]]
        parents2 = parents[parents_list[n_particles:]]
        pp1 = obj_parents[parents_list[:n_particles]]
        pp2 = obj_parents[parents_list[n_particles:]]
        pp2 = pp2 / (pp1+pp2)
        pp2 = pp2[(..., ) + (None,)*3]

        mask_a = torch.empty_like(parents1).uniform_() > pp2
        mask_b = ~mask_a
        parents1[mask_a] = 0.0
        parents2[mask_b] = 0.0
        children = parents1 + parents2

        # add some mutations to children and genrate test set for next generation
        children = mutation(x_seed, children, eps, p=0.2)
        adv_images = torch.cat([children,parents], dim=0)
    # print("test1", np.unique(adv_images))
    obj, loss, op_loss = fitness_score(x_seed, y_seed, adv_images, model, local_op,alpha)
    sorted, indices = torch.sort(loss, dim=-1, descending=True)
    return adv_images[indices[:10]], loss[indices[:10]], op_loss[indices[:10]]




def test_model(vae, model, dataset, batch_size, latent_dim, data_config, output_dir, eps, n_seeds, local_op, cuda, index):
    # output_dir = "./svhn_output_check"
    torch.manual_seed(0)
    vae.eval()
    model.eval()
    # model_keras = keras.models.load_model("/data/c/tianmeng/wlt/cifar10_vgg_model.194.h5")
    # model_keras = svhn_vgg16()
    # model_keras = keras.models.load_model("/data/c/tianmeng/wlt/fm_lenet5.h5")
    # model_keras = keras.models.load_model("/data/c/tianmeng/wlt/lenet5_softmax.h5")
    # n_channel  = data_config['channels']
    # img_size = data_config['size']
    # n_class = data_config['classes']

    # # dataset = dataset[:2000]
    # n = len(dataset)
    # n = 2000
    # print(type(dataset), n)
    # data_loader = utils.get_data_loader(dataset, batch_size, cuda = cuda)

    # grad_norm = []
    # x_act_dist = []

    # # Get data into arrays for convenience
    # if cuda:
    #     x_test = torch.zeros(n, n_channel, img_size, img_size).cuda()
    #     y_test = torch.zeros(n, dtype = int).cuda()
    #     y_pred = torch.zeros(n, dtype = int).cuda()
    #     x_mu = torch.zeros(n, latent_dim).cuda()
    #     x_std = torch.zeros(n, latent_dim).cuda()

    # else:
    #     x_test = torch.zeros(n, n_channel, img_size, img_size)
    #     y_test = torch.zeros(n, dtype = int)
    #     y_pred = torch.zeros(n, dtype = int)
    #     x_mu = torch.zeros(n, latent_dim)
    #     x_std = torch.zeros(n, latent_dim)

    # print(index)
    # for idx, (data, target) in tqdm(enumerate(data_loader)):
    #     # print(idx)
    #     if index*20<= idx and idx < (index+1)*20: 
    #         print("test",idx)
    #         data_numpy = data.numpy()
    #         print(data_numpy.shape)
    #         # print(np.unique(data_numpy))
    #         target_numpy = target.numpy()
    #         data_numpy = np.transpose(data_numpy, (0,2,3,1))
    #         # data_numpy = data_numpy *255
    #         if cuda:
    #             data, target = data.float().cuda(), target.long().cuda()
    #         else:
    #             data, target = data.float(), target.long()
    #         if len(target.size()) > 1:
    #             target = torch.argmax(target, dim=1)
    #         # print(model_keras.evaluate(data_numpy, target_numpy))
    #         # grad_batch = cal_gradient(model,data,target)
            
    #         grad_batch_keras = cal_gradient_keras(model_keras, data_numpy, target_numpy)
    #         # print(grad_batch_keras.shape)
    #         # break
    #         # grad_norm.append(grad_batch)
    #         grad_norm.append(grad_batch_keras)

    #         with torch.no_grad():
    #             mu, log_var = vae.encode(data)
    #             # hidden_act = model.hidden_act(data)
    #             hidden_act_keras = extract_data_features(model_keras, data_numpy, -3)
    #             # target_pred = torch.argmax(model(data/2+0.5), dim=1)
    #             target_pred = np.argmax(model_keras.predict(data_numpy/2+0.5), axis=1)

    #             # x_test[(idx * batch_size):((idx + 1) * batch_size), :, :, :] = data
    #             # y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
    #             # y_pred[(idx * batch_size):((idx + 1) * batch_size)] = torch.tensor(target_pred)
    #             # x_mu[(idx * batch_size):((idx + 1) * batch_size), :] = mu
    #             # x_std[(idx * batch_size):((idx + 1) * batch_size), :] = torch.exp(0.5 * log_var)
    #             tmp_idx = idx - index*20
    #             x_test[(tmp_idx * batch_size):((tmp_idx + 1) * batch_size), :, :, :] = data
    #             y_test[(tmp_idx * batch_size):((tmp_idx + 1) * batch_size)] = target
    #             y_pred[(tmp_idx * batch_size):((tmp_idx + 1) * batch_size)] = torch.tensor(target_pred)
    #             x_mu[(tmp_idx * batch_size):((tmp_idx + 1) * batch_size), :] = mu
    #             x_std[(tmp_idx * batch_size):((tmp_idx + 1) * batch_size), :] = torch.exp(0.5 * log_var)
                
    #             hidden_act_keras = torch.Tensor(hidden_act_keras)
    #             x_act_dist.append(hidden_act_keras)
        
    # # print("pass")
    # # print(hidden_act_keras)
    # grad_norm = torch.cat(grad_norm, dim=0)
    # x_act_dist = torch.cat(x_act_dist, dim=0)
    # # print(grad_norm.shape)
    # # print(x_act_dist.shape)
    # # print(y_test.shape)
    # print(Counter(y_test.cpu().numpy()))
    # indices = torch.where(y_pred==y_test)
    # x_test = x_test[indices] 
    # y_test = y_test[indices]
    # x_mu = x_mu[indices]
    # x_std = x_std[indices]
    # grad_norm = grad_norm[indices]
    # x_act_dist = x_act_dist[indices]

    # # indices = torch.randperm(len(x_test))[:10000]
    # # x_test = x_test[indices] 
    # # y_test = y_test[indices]
    # # x_mu = x_mu[indices]
    # # x_std = x_std[indices]
    # # grad_norm = grad_norm[indices]
    # # x_act_dist = x_act_dist[indices]



    # # # # ################################################################################################### 
    # # start = time.time()     
    # # print()
    # # print('Start to test the model!')
    # # print()
    # # print('Dataset:', model.label)
    # # print('No. of test seeds:', n_seeds)
    # # print('Total No. in test set:', len(x_mu))
    # # print('Norm ball radius:', eps)


    # kde = GaussianKDE(x_mu, x_std)
    # pd = kde.score_samples(x_mu)
  
    # grad_aux = utils.min_max_scale(grad_norm.cpu())

    # aux_inf = pd * grad_aux
    # sorted, indices = torch.sort(aux_inf, dim=-1, descending=True)

    # x_seeds = x_test[indices[:n_seeds]]
    # y_seeds = y_test[indices[:n_seeds]]
    # x_seeds = x_seeds.cpu().numpy()
    # y_seeds = y_seeds.cpu().numpy()
    # x_seeds = np.transpose(x_seeds,(0,2,3,1))
    # print(Counter(y_seeds))
    # print(x_seeds.shape)
    # print(y_seeds.shape)
    # # x_seeds = x_seeds.reshape(len(x_seeds),-1)
    # # x_seeds = 
    # # print(x_seeds.shape)
    # # print(np.unique(x_seeds))
    # np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/temp_data_%s.npy"%index, x_seeds)
    # np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/temp_truth_%s.npy"%index, y_seeds)
    # # for i in range(10):
    # #     temp_data = x_seeds[np.where(y_seeds==i)[0]]
    # #     temp_truth = y_seeds[np.where(y_seeds==i)[0]] 
    # #     np.save("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/svhn/class_%s_seed.npy"%i, temp_data)
        
        
    # # # print(np.unique(x_seeds))
    # # # print(type(x_seeds))

    # seed_select_index = 7
    for seed_select_index in [0]:
        # x_seeds = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/class_%s_seed.npy"%seed_select_index)
        x_seeds = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/seeds.npy")
        # x_seeds = x_seeds.reshape(-1,28,28)
        # x_seeds = x_seeds.reshape(-1,28,28,1)
        x_seeds = x_seeds / 255.
        # x_seeds = svhn_preprocessing(x_seeds)
        # if seed_select_index == 0:
        #     truth_label = 9
        # else:
        #     truth_label = seed_select_index - 1
        # y_seeds = np.ones(len(x_seeds), dtype=np.int32) * seed_select_index
        y_seeds = np.load("/home/dltest/tianmeng/wlt/HDA-Testing-main/select_seeds/mnist/truth.npy")
        x_seeds = np.transpose(x_seeds, (0,3,1,2))
        print(x_seeds.shape)
        x_seeds = torch.tensor(x_seeds)
        y_seeds = torch.tensor(y_seeds)

        # model = get_svhn_model()
        # model.load_weights("./svhn_vgg16_weight.h5")
        # sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # model = keras.models.load_model("/data/c/tianmeng/wlt/lenet5_softmax.h5")
        model = mnist_lenet4()
        # print(model.evaluate(x_seeds, keras.utils.to_categorical(y_seeds, 10)))
       

        # # ###################################################################################################
        # test case generation with Two-step Genetic Algorithm
        # GA settings
        # model = keras.models.load_model("../dissector/cifar10_vgg_model.194.h5")
        n_particles = 1000
        n_mate = 20
        max_itr = 100
        alpha = 1.00 
        # local_op = "mnist_hdaselect"
        save_dir = output_dir+'/hdaselece_HDA_'+ local_op
        save_seeds_dir = save_dir+'/test_seeds_mnist_%s'%seed_select_index
        save_aes_dir = save_dir+'/AEs_mnist_%s'%seed_select_index

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_seeds_dir):
            os.makedirs(save_seeds_dir)
        if not os.path.exists(save_aes_dir):
            os.makedirs(save_aes_dir)

        count = 0
        adv_count = 0
        test_set = []
        adv_set = []
        test_label = []
        loss_set = []
        seed_set = []

        with torch.no_grad():
            for idxx, (x_seed, y_seed) in tqdm(enumerate(zip(x_seeds,y_seeds))):
                # print(idxx)
                # if idxx >=91:
                # x_seed = x_seed /2 + 0.5
                # print(1, np.unique(x_seed))
                # save test seeds to files
                
                # vutils.save_image(
                # x_seed,
                # save_seeds_dir+'/{no}_{label}.png'.format(no = count, label=y_seed.item()),
                # normalize=False)
                
                # print(2, np.unique(x_seed))
                # start running GA on seeds input
                torch.cuda.empty_cache()
                # print(3, np.unique(x_seed))
                # print(x_seed)
                ae, loss, op_loss = two_step_ga(model, x_seed, y_seed, eps, local_op, n_particles = n_particles, n_mate = n_mate, max_itr = max_itr, alpha = alpha)
                # print(4, np.unique(ae))
                test_set.append(ae.cpu())
                test_label.append(torch.stack(10*[y_seed]).cpu())

                idx = torch.where(loss>=0)[0]
                if len(idx)>0:
                    ae = ae[idx[0]]
                    ae_loss = loss[idx[0]]
                    # ae_pred = torch.argmax(model(ae.unsqueeze(0)), dim=1)
                    # a = model(ae.unsqueeze(0))
                    # c = extract_data_features(model, (ae.unsqueeze(0)).numpy(), -2)
                    # b = torch.tensor(extract_data_features(model, (ae.unsqueeze(0)).numpy(), -2))
                    print(ae.shape)
                    temp_ae = ae.numpy()
                    # temp_ae = temp_ae.reshape(28,28)
                    # temp_ae = temp_ae.reshape(28,28,1)
                    temp_ae = np.transpose(temp_ae, (1,2,0))
                    print(temp_ae.shape)
                    temp_ae = np.array([temp_ae])
                    # a = extract_data_features(model, temp_ae, -2)
                    ae_pred = np.argmax(extract_data_features(model, temp_ae, -2))

                    adv_set.append(ae.unsqueeze(0))
                    seed_set.append(x_seed.unsqueeze(0))
                    loss_set.append(ae_loss.cpu().unsqueeze(0))
                    adv_count += 1
                    # print("5", np.unique(ae))
                    np.save(save_aes_dir + "/%s_%s_%s.npy"%(count, y_seed.item(), ae_pred.item()), ae)
                    # vutils.save_image(
                    #     ae,
                    #     save_aes_dir+'/{no}_{label}_{pred}.png'.format(no = count, label=y_seed.item(), pred = ae_pred.item()),
                    #     normalize=False)



                count += 1




    # # # torch.save([test_set,test_label],'AEs_fine_tune_10/'+ model.label +'_hda.pt')


    # # # calculate l_inf norm beween seeds and AEs
    # # adv_set = torch.cat(adv_set)
    # # seed_set = torch.cat(seed_set)
    # # loss_set = torch.cat(loss_set)
    # # print('################# Local AEs Perceptual Quality ###########################')
    # # epsilon = torch.norm(adv_set-seed_set,p=float('inf'),dim=(1,2,3))
    # # print('Avg. Perturbation Amount:',torch.mean(epsilon).item())
    
    # # model_label = "mnist"
    
    # # # calculate FID between seeds and AEs
    # # # if model.label == 'mnist' or model.label == 'FashionMnist':
    # # if model_label == 'mnist' or model_label == 'FashionMnist':
    # #     with torch.no_grad():
    # #         seed_mu, seed_var = vae.encode((seed_set-0.5)*2)
    # #         adv_mu, adv_var = vae.encode((adv_set-0.5)*2)
    # #     fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    # # else:
    # #     # prepare for calculating fid
    # #     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    # #     inception = InceptionV3([block_idx]).cuda()

    # #     with torch.no_grad():
    # #         adv_mu = inception(adv_set)[0]
    # #         seed_mu = inception(seed_set)[0]
    # #         adv_mu= torch.flatten(adv_mu, start_dim = 1)
    # #         seed_mu = torch.flatten(seed_mu, start_dim = 1)
    # #     fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))

    # # end = time.time()

    # # print('FID (adv.): %.3f' % fid)
    # # print('Attack success rate:', adv_count/n_seeds)
    # # print('Avg. prediction loss of AEs:', torch.mean(loss_set).item())
    # # print('Elapsed time:',end - start)

   
    # # # generate a test report
    # # f = open(save_dir+ "/test_report.txt", "w")
    # # f.write('Dataset: {}\n'.format(model.label))
    # # f.write('Total No. of test seeds: {}\n'.format(n_seeds))
    # # f.write('Norm ball radius: {}\n'.format(eps))
    # # f.write('################# Global Seeds Probability Density ###########################\n')
    # # f.write('KDE probability density: {}\n'.format((sum(op)/sum(pd)).item()))
    # # f.write('Rand. probability density: {}\n'.format((sum(op_rand)/sum(pd)).item()))
    # # f.write('################# Local AEs Perceptual Quality ###########################\n')
    # # f.write('Avg. Perturbation Amount: {}\n'.format(torch.mean(epsilon).item()))
    # # f.write('FID (adv.): {}\n'.format(fid))
    # # f.write('Detect AEs success rate: {}\n'.format(adv_count/n_seeds))
    # # f.write('Avg. prediction loss of AEs: {}\n'.format(torch.mean(loss_set).item()))
    # # f.write('Elapsed time: {}\n'.format(end - start))
    # # f.close()

    



    