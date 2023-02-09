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
import torchattacks
import time

def attack_model(vae,model,dataset,eps,cuda, batch_size, latent_dim, data_config):

    torch.manual_seed(0)
    vae.eval()
    model.eval()


    start = time.time()

    # atk = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=10)
    atk = torchattacks.FGSM(model, eps=eps)
    # atk = torchattacks.AutoAttack(model, norm='Linf', eps=eps, version='standard', n_classes=10, seed=None, verbose=False)

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,batch_size=100)
    x_seeds, y_seeds = data_loader.__iter__().next()

    if cuda:
        x_seeds = x_seeds.cuda()
        y_seeds = y_seeds.cuda()

    x_seeds = x_seeds /2 + 0.5


    test_cases = atk(x_seeds, y_seeds)
    y_pred = model(test_cases)
    pred_label = torch.argmax(y_pred, dim = 1)

    

    adv_idx = torch.where(y_seeds != pred_label)
    adv_images = test_cases[adv_idx]
    adv_seeds = x_seeds[adv_idx]
    y_adv = y_pred[adv_idx]
    adv_label = y_seeds[adv_idx]
    adv_label = adv_label[:,None]

    y_diff = y_adv- torch.gather(y_adv,1,adv_label)
    y_diff, _ = y_diff.max(dim=1)

    epsilon = torch.norm((adv_images-adv_seeds)/2,p=float('inf'),dim=(1,2,3))


    if model.label == 'mnist' or model.label == 'FashionMnist':
        with torch.no_grad():
            adv_mu, log_var = vae.encode((test_cases-0.5)*2)
            seed_mu, log_var = vae.encode((x_seeds-0.5)*2)
            fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    else:
        # prepare for calculating fid
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception = InceptionV3([block_idx]).cuda()

        with torch.no_grad():
            adv_mu = inception(adv_images)[0]
            seed_mu = inception(x_seeds)[0]
            adv_mu= torch.flatten(adv_mu, start_dim = 1)
            seed_mu = torch.flatten(seed_mu, start_dim = 1)
        fid = calculate_fid(np.array(adv_mu.cpu()),np.array(seed_mu.cpu()))
    
    end = time.time()

    pg = test_density(vae, model, dataset,data_loader,batch_size, latent_dim, data_config, cuda, test_cases)

    print('FID (adv.): %.3f' % fid)
    print('perturb amount is ',torch.mean(epsilon).item())
    print('attack success rate', len(adv_images)/len(x_seeds))
    print('prediction loss of adv', torch.mean(y_diff).item())
    print('kde probability density is ', pg.item())
    print('elapsed time',end - start)


def test_density(vae, model, dataset, data_loader,batch_size, latent_dim, data_config, cuda, test_cases):

    torch.manual_seed(0)
    vae.eval()
    model.eval()

    n = len(dataset)

    # Get data into arrays for convenience
    if cuda:
        y_test = torch.zeros(n, dtype = int).cuda()
        y_pred = torch.zeros(n, dtype = int).cuda()
        x_mu = torch.zeros(n, latent_dim).cuda()
        x_std = torch.zeros(n, latent_dim).cuda()

    else:
        y_test = torch.zeros(n, dtype = int)
        x_mu = torch.zeros(n, latent_dim)
        x_std = torch.zeros(n, latent_dim)


    
    for idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = data.float().cuda(), target.long().cuda()
        else:
            data, target = data.float(), target.long()

        if len(target.size()) > 1:
            target = torch.argmax(target, dim=1)


        with torch.no_grad():
            mu, log_var = vae.encode(data)
            target_pred = torch.argmax(model(data/2+0.5), dim=1)

            y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
            y_pred[(idx * batch_size):((idx + 1) * batch_size)] = target_pred
            x_mu[(idx * batch_size):((idx + 1) * batch_size), :] = mu
            x_std[(idx * batch_size):((idx + 1) * batch_size), :] = torch.exp(0.5 * log_var)


    indices = torch.where(y_pred==y_test)
    
    x_mu = x_mu[indices]
    x_std = x_std[indices]


    # # # ###################################################################################################   
    # 
    mu_set, log_var_set = vae.encode((test_cases-0.5)*2)  
    with torch.no_grad():
        x_mu = torch.cat([x_mu,mu_set]) 
        x_std = torch.cat([x_std, torch.exp(0.5 * log_var_set)])

    kde = GaussianKDE(x_mu, x_std)
    pd = kde.score_samples(x_mu)

    op = pd[-len(mu_set):]

    return sum(op)/sum(pd)



    