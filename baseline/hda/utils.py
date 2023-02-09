import os
import os.path
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from torchvision import transforms
import numpy as np
import time
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import keras
from keras import backend, losses
import tensorflow as tf
from get_model import svhn_preprocessing, cifar_preprocessing

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


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle = False,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def save_checkpoint_adv(model,mode,model_dir, epoch):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint_adv(model, model_dir,mode):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']


def load_checkpoint(model, model_dir,cuda):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch

def get_nearest_oppo_dist(X, y, norm, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=p, n_jobs=12).fit(X[y != yi])

    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y == yi], n_neighbors=1)
        ret[np.where(y == yi)[0]] = dist[:, 0]

    return nns, ret

def filter_celeba(dataset):
    # drop unrelated attr
    attr = dataset.attr
    attr_names = dataset.attr_names[:40]
    new_attr_names = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    mask_attr = torch.tensor([True if x in new_attr_names else False for x in attr_names]) 
    dataset.attr = attr[:,mask_attr]
    dataset.attr_names = new_attr_names
    # keep only 1 attr instance and drop others
    mask_id = torch.sum(dataset.attr, dim = 1) == 1
    dataset = torch.utils.data.Subset(dataset, torch.where(mask_id)[0])
    return dataset

def cal_robust(x_sample, x_class, model, CUDA, grey_scale,sigma):

    if grey_scale:
        robustness_stat = greyscale_multilevel_uniform
    else:
        robustness_stat = multilevel_uniform

    # sigma = 0.1
    rho = 0.1
    debug= True
    stats=False
    count_particles = 1000
    count_mh_steps = 200

    print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

    def prop(x):
      y = model(x)
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff #.max(dim=1)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = robustness_stat(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                              count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)
      print('---------------------------------')

    return lg_p

def cal_gradient(model,images,labels):
    loss = nn.CrossEntropyLoss()
    images.requires_grad = True
    outputs = model(images)
    cost = loss(outputs, labels)
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    print(grad.shape)
    print(type(grad))
    grad_norm = torch.norm(grad,p = np.inf, dim = [1,2,3])
    # print(grad_norm.shape)
    return grad_norm

# def cal_gradient_keras(model, images, labels):
#     outputs = model.predict(images)
#     print(outputs.shape)
#     loss = losses.categorical_crossentropy(outputs, keras.utils.to_categorical(labels, 10))
#     images = tf.convert_to_tensor(images)
#     for i in range(len(outputs)):
#         gradient = backend.gradients(loss[i], images[i])
#         print(gradient)
#     return gradient
    
def cal_gradient_keras(model, images, labels):
    # images = images *255
    # images = cifar_preprocessing(images)
    y_pred = model.output
    loss = losses.categorical_crossentropy(keras.utils.to_categorical(labels, 10), y_pred)
    gradient = backend.gradients(loss, model.input)
    gradient = gradient[0] 
    
    sess = backend.get_session()
    gradient = sess.run(gradient, feed_dict={model.input : images})
    gradient = np.transpose(gradient, (0,3,1,2))
    gradient = torch.tensor(gradient)
    grad_norm = torch.norm(gradient, p = np.inf, dim = [1,2,3])
    return grad_norm

def mutation(x_seed, adv_images, eps, p):
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.003)
    mask = torch.empty_like(adv_images).uniform_() > p 
    delta[mask] = 0.0
    delta = adv_images + delta - x_seed
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0.0, max=1.0).detach()
    return adv_images

def pred_loss(x,x_class,model):
    print(x.shape)
    print(x_class)
    with torch.no_grad():
      y = model(x)
    #   print(y.shape)
    #   print(type(y))
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff 
  
def pred_loss_keras(x, x_class, model):
    x = x.numpy()
    # x = x.reshape(len(x), 28, 28)
    # x = x.reshape(len(x), 28, 28, 1)
    x = np.transpose(x, (0,2,3,1))
    y = extract_data_features(model, x, -2)
    y = torch.tensor(y)
    y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
    y_diff, _ = y_diff.max(dim=1)
    return y_diff
    

def cal_dist(x, x_a, model):
    model.eval()
    act_a = model(x_a)[0]
    act = model(x)[0]
    act_a = torch.flatten(act_a, start_dim = 1)
    act = torch.flatten(act, start_dim = 1)
    mse = calculate_fid(act, act_a)
    return mse

def mse(x,x_a):
    loss = (x_a - x)**2
    return torch.mean(loss,dim=[1,2,3])

def psnr(x,x_a):
    mse_loss = torch.mean((x_a - x) ** 2, dim=[1,2,3])
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

def ms_ssim_module(x,x_a):
    x, x_a = torch.broadcast_tensors(x, x_a)
    ms_ssim_val = SSIM(data_range=1, size_average=False, channel=x.shape[-3])
    return ms_ssim_val(x,x_a)

def min_max_scale(x):
    return (x-x.min())/(x.max()-x.min())

def fitness_score(x,y,x_a,model,local_op,alpha):
    loss = pred_loss_keras(x_a,y,model)

    if local_op == 'None':
        op = None
        obj = min_max_scale(loss)
        return obj, loss, op

    elif local_op == 'mse':
        op = -mse(x,x_a)
    elif local_op == 'psnr':
        op = psnr(x,x_a)
    elif local_op == 'ms_ssim':
        op = ms_ssim_module(x,x_a)
    else:
        raise Exception("Choose the support local_p from None, mse, psnr, ms_ssim")


    if torch.sum(loss>0)/len(loss) < 0.6 :
        obj = min_max_scale(loss)
    else:
        obj = min_max_scale(loss) +  alpha * min_max_scale(op)
    # obj = min_max_scale(loss) +  alpha *  min_max_scale(op)

    return obj, loss, op

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*torch.log2(q[i]) for i in range(len(p))])

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

    

    
    

    




