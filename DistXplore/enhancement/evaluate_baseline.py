import keras
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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


score_list = []
# for model_prefix in ["high_coverage", "bim", "pgd", "cw","kmnc", "nbc"]:
for model_prefix in ["high_coverage"]:
    model = keras.models.load_model("/data/c/tianmeng/wlt/retrain_models/cifar_resnet/%s_ft_model_20_all.h5"%model_prefix)
    temp_score_list = []
    for tech in ["bim", "pgd", "cw"]:
    # for tech in ["kmnc", "nbc"]:
        temp_data = np.load("/data/c/tianmeng/wlt/evaluate_data/cifar/%s/data.npy"%tech)
        temp_truth = np.load("/data/c/tianmeng/wlt/evaluate_data/cifar/%s/ground_truth.npy"%tech)
        shuffle_index = np.random.permutation(len(temp_data))
        temp_data = temp_data[shuffle_index]
        temp_truth = temp_truth[shuffle_index]
        temp_data = temp_data[:1000]
        temp_truth = temp_truth[:1000]
        # temp_data = temp_data[:500]
        # temp_truth = temp_truth[:500]
        if tech in ["kmnc", "nbc", "hda", "vae"]:
            # temp_data = temp_data /255.
            temp_data = cifar_preprocessing(temp_data)
        temp_truth = keras.utils.to_categorical(temp_truth, 10)
        score = model.evaluate(temp_data, temp_truth)[1]
        temp_score_list.append(score)
        print(model_prefix, tech, score)
    score_list.append(temp_score_list)
print(score_list)
# np.save("/data/c/tianmeng/wlt/enhancement_results/ga_svhn_acc.npy", score_list)