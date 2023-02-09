#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from xmlrpc.client import boolean
from numpy.core.fromnumeric import argmax
import torch
import cv2
import torchvision
from distutils import util
import numpy as np

import os.path
from utils import calculate_fid
from torchvision.datasets import celeba
from model.vae_celeba import VAE_celeba
from model.vae import VAE
from model.celeba import MobileNet
from model.mnist import mnist
from model.fashionmnist import FashionCNN
from model.svhn import svhn
from model.cifar10 import resnet20
from data import TRAIN_DATASETS, DATASET_CONFIGS, TEST_DATASETS
from vae_train import train_vae_model
from train import train_model
from adv_train import adv_train_model
from global_robustness import robustness_eval, process_data
from test import test_model
from adv_attack import attack_model
from adv_coverage import coverage_test
import utils


parser = argparse.ArgumentParser('HDA Testing Pytorch Implementation')
parser.add_argument('--dataset', default='mnist', choices=list(TRAIN_DATASETS.keys()))
parser.add_argument('--no_seeds', default = 10, dest='no_seeds',type=int)
parser.add_argument('--local_p', default = 'mse', choices=['None','mse','psnr','ms_ssim'])
parser.add_argument('--train', default = False, dest='train',type=util.strtobool)
parser.add_argument('--vae_train', default = False, dest='vae_train',type=util.strtobool)
parser.add_argument('--adv_train', default = False, dest='adv_train')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--sample-size', type=int, default=20)
parser.add_argument('--lr', type=float, default = 1e-04)
parser.add_argument('--weight-decay', type=float, default = 0)
parser.add_argument('--resume', default = False,type=util.strtobool)
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
parser.add_argument('--sample-dir', type=str, default='samples')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--index', type=int)

# mnist 
# FashionMnist 
# svhn 
# cifar10 
# celeba 

if __name__ == '__main__':
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    # cuda = False
    index = args.index
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_dataset = TRAIN_DATASETS[args.dataset]
    test_dataset = TEST_DATASETS[args.dataset]
    
    # print(np.unique(train_dataset))
    # print(np.unique(test_dataset))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)


    if args.dataset == 'celeba':
        
        model = MobileNet(num_classes=dataset_config['classes'], label = args.dataset)

        latent_dim = 32
        hidden_dim = None
        eps = 0.05

        vae = VAE_celeba(
        label = args.dataset,
        image_size = dataset_config['size'],
        in_channels=dataset_config['channels'],
        latent_dim = latent_dim,
        hidden_dims = hidden_dim
        )
    
        if not args.vae_train:
            train_dataset = utils.filter_celeba(train_dataset)
            test_dataset = utils.filter_celeba(test_dataset)
    else:

        if args.dataset == 'mnist':
            latent_dim = 8
            hidden_dim = 256
            eps = 0.1
            model = mnist(num_classes=dataset_config['classes'], label = args.dataset)

        if args.dataset == 'FashionMnist':
            latent_dim = 4
            hidden_dim = 128
            eps = 0.08
            model = FashionCNN(num_classes=dataset_config['classes'], label = args.dataset)

        if args.dataset == 'svhn':
            latent_dim = 4
            hidden_dim = 256
            eps = 0.03
            model = svhn(num_classes=dataset_config['classes'], data_name= args.dataset)

        if args.dataset == 'cifar10':
            latent_dim = 8
            hidden_dim = 256
            eps = 0.03
            model = resnet20(num_classes=dataset_config['classes'], label = args.dataset)


        vae = VAE(
        label=args.dataset,
        image_size=dataset_config['size'],
        input_dim=dataset_config['channels'],
        dim=hidden_dim,
        z_dim=latent_dim,
        )
        # latent_dim *= 4


    # move the model parameters to the gpu if needed.
    if cuda:
        vae.cuda()
        model.cuda()
    # run a test or a training process.
    if args.train:
        if args.vae_train:
            if not os.path.exists(args.sample_dir):
                os.makedirs(args.sample_dir)

            train_vae_model(
                vae, train_dataset=train_dataset,
                test_dataset=test_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                cuda=cuda
            )
        elif args.adv_train:
            adv_train_model(
                model, train_dataset=train_dataset,
                test_dataset=test_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                checkpoint_dir=args.checkpoint_dir,
                cuda=cuda
            )
        else:
            train_model(
                model, train_dataset=train_dataset,
                test_dataset=test_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                cuda=cuda
            )

    else:
        utils.load_checkpoint(vae, args.checkpoint_dir, cuda)
        # utils.load_checkpoint(model, args.checkpoint_dir, cuda)
        
        

 
        # main experiments by utlizing distribution for generating AEs
        test_model(vae, model, train_dataset, args.batch_size, latent_dim, dataset_config, args.dataset+ '_' +args.output_dir, eps, args.no_seeds, args.local_p, cuda, index)

        #################### compare with pgd and coverage guided testing ####################################
        # # generate AEs by PGD
        # attack_model(vae, model, train_dataset, eps, cuda, args.batch_size, latent_dim, dataset_config)

        # # generate AEs by Coverage Guided Tesing
        # coverage_test(vae, model, test_dataset, eps,cuda)

        #################### RQ4 adversarial fine-tuning #####################################################
        # # test global robustness of model
        # robustness_eval(vae, model, train_dataset, args.batch_size, latent_dim, dataset_config, args.dataset+ '_' +args.output_dir, eps, args.no_seeds, args.local_p, cuda)

        # process_data(vae, model, train_dataset, args.batch_size, latent_dim, dataset_config, args.dataset+ '_' +args.output_dir, eps, args.no_seeds, args.local_p, cuda)



    
        
       
        


       
