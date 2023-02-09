from numpy.lib.function_base import append
import utils
import torch
import torch.nn as nn
from model.inception import InceptionV3
from GaussianKDE import GaussianKDE
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import numpy as np
from utils import get_nearest_oppo_dist, cal_robust, cal_gradient, fitness_score, mutation, pred_loss
import numpy as np
from utils import calculate_fid
import time
import copy
from neuron_coverage import *

def coverage_test(vae, model, dataset,eps,cuda):

    model.eval()
    vae.eval()
    torch.manual_seed(0)

    start = time.time()

    # test case generation
    max_itr = 10
    adv_set = []

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,batch_size=128)
    x_seeds, y_seeds = data_loader.__iter__().next()

    if cuda:
        x_seeds = x_seeds.cuda()
        y_seeds = y_seeds.cuda()

    x_seeds = x_seeds /2 + 0.5



    test_set = []
    adv_set = []
    adv_loss = []
    # initialize the population

    for i in range(max_itr):
        coverage_record = []
        candidate_test_cases = []
        adv_test_cases = []
        adv_loss_cases = []
        
        for j in range(10):
            delta = torch.empty_like(x_seeds).normal_(mean=0,std=0.6)
            delta = torch.clamp(delta, min=-eps, max=eps)
            test_cases = torch.clamp(x_seeds + delta, min=0, max=1).detach()
            y_pred = model(test_cases)
            pred = torch.argmax(y_pred,dim = 1)
            adv_idx = torch.where(pred != y_seeds)
            adv_cases = test_cases[adv_idx]
            if len(adv_cases) != 0:
                y_adv = y_pred[adv_idx]
                adv_label = y_seeds[adv_idx]
                adv_label = adv_label[:,None]
                y_diff = y_adv- torch.gather(y_adv,1,adv_label)
                y_diff, _ = y_diff.max(dim=1)
            else:
                y_diff = torch.tensor([]).cuda()


            test_set_copy = copy.deepcopy(test_set)
            test_set_copy.append(test_cases)
            covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, torch.cat(test_set_copy,dim=0) , 0.0)
            coverage_record.append(neuron_coverage_000)
            candidate_test_cases.append(test_cases)
            adv_test_cases.append(adv_cases)
            adv_loss_cases.append(y_diff)
            

        idx = np.argmax(coverage_record)
        test_set.append(candidate_test_cases[idx])
        adv_set.append(adv_test_cases[idx])
        adv_loss.append(adv_loss_cases[idx])

    test_set = torch.cat(test_set,dim=0)
    adv_set = torch.cat(adv_set, dim=0)
    adv_loss = torch.cat(adv_loss, dim=0)

    if model.label == 'mnist' or model.label == 'FashionMnist':
        with torch.no_grad():
            adv_mu, log_var = vae.encode((test_set-0.5)*2)
            seed_mu, log_var = vae.encode((x_seeds-0.5)*2)
            fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    else:
        # prepare for calculating fid
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception = InceptionV3([block_idx]).cuda()

        with torch.no_grad():
            adv_mu = inception(test_set)[0]
            seed_mu = inception(x_seeds)[0]
            adv_mu= torch.flatten(adv_mu, start_dim = 1)
            seed_mu = torch.flatten(seed_mu, start_dim = 1)
        fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))


    end = time.time()

    print('FID (adv.): %.3f' % fid)
    print('neuron coverage rate is', np.max(coverage_record))
    print('AE detect success rate:',len(adv_set)/len(test_set))
    print('pred loss:',torch.mean(adv_loss).item())
    print('elapsed time',end - start)

    


