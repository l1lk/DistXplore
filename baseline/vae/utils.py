import random
from collections import defaultdict
import os
import glob
import cv2
import numpy as np
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
import scikitplot as skplt
import matplotlib.pyplot as plt
from PIL import Image
import keras
from sklearn.metrics import precision_score, recall_score, f1_score
import math 

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False


def cumulative_neuron_coverage(model_layer_dict1, model_layer_dict2, model_layer_dict3):
    for (layer_name, index), v in model_layer_dict1.items():
        model_layer_dict3[(layer_name, index)] = v or model_layer_dict2[(layer_name, index)]


def neurons_covered_uncommon(model_layer_dict1, model_layer_dict2):
    result = []
    #dict1 are valid tests and dict2 are invalid
    for (layer_name, index), v in model_layer_dict1.items():
        if (not v) and model_layer_dict2[(layer_name, index)]:
            result.append((layer_name, index))
    return result

def neuron_not_covered(model_layer_dict1):
    result = []
    for (layer_name, index), v in model_layer_dict1.items():
        if (not v):
            result.append((layer_name, index))
    return result



def delete_files_from_dir(dirPath, ext):
    # eg input = /tmp/*.txt
    fileFormat = dirPath + '*.' + ext
    files = glob.glob(fileFormat)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


# This api is for sampling from latent space of VAE
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Logic for calculating reconstruction probability
def reconstruction_probability(decoder, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    reconstructed_prob = np.zeros((X.shape[0],), dtype='float32')
    L = 10
    for l in range(L):
        # print(l)
        sampled_zs = sampling([z_mean, z_log_var])
        mu_hat, log_sigma_hat = decoder.predict(sampled_zs, steps=1)
        log_sigma_hat = np.float64(log_sigma_hat)
        sigma_hat = np.exp(log_sigma_hat) + 0.00001

        loss_a = np.log(2 * np.pi * sigma_hat)
        loss_m = np.square(mu_hat - X) / sigma_hat
        reconstructed_prob += -0.5 * np.sum(loss_a + loss_m, axis=1)
    reconstructed_prob /= L
    return reconstructed_prob


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, vae):
    x_target_orig = np.clip(x_target_orig, 0, 1)
    x_target_orig = np.reshape(x_target_orig, (-1, 28*28))
    x_target = np.reshape(x_target_orig, (-1, 28, 28, 1))
    # print("a.1")
    z_mean, z_log_var, _ = vae.get_layer('encoder').predict(x_target,
                                                            batch_size=128)
    # print("a.2")
    reconstructed_prob_x_target = reconstruction_probability(vae.get_layer('decoder'), z_mean, z_log_var, x_target_orig)
    # print("a.3")
    return reconstructed_prob_x_target
    

# checks whether a test input is valid or invalid
#Returns true if invalid
def isInvalid(gen_img, vae, vae_threshold):
    print("a")
    gen_img_density = calculate_density(gen_img, vae)
    # print("b")
    if gen_img_density < vae_threshold or math.isnan(gen_img_density):
        return True
    else:
        return False
