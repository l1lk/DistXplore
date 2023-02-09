import copy
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from model.inception import InceptionV3
from GaussianKDE import GaussianKDE
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import torch.utils.data as data_utils
import torchvision.utils as vutils
import numpy as np
from utils import get_nearest_oppo_dist, cal_gradient, fitness_score, mutation, pred_loss
import numpy as np
from utils import calculate_fid
import torchattacks
import time

def test(data_loader, model,cuda):
    model.eval()
    pred_id = []
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        for batch_index, (x, y, z) in data_stream:
            # prepare data on gpu if needed
            if cuda:
                x = x.to('cuda')
                y = y.to('cuda')
                z = z.to('cuda')
                atk_succes = torch.ones(len(x)).cuda()
            else:
                atk_succes = torch.ones(len(x))

            if len(y.size()) > 1:
                real_labels = torch.argmax(y, dim=1)
            else:
                real_labels = y

            result = model(x)
            loss = criterion(result, real_labels)
            running_loss += loss.item()

            pred_labels = torch.argmax(result, dim=1)
            atk_succes[torch.where(pred_labels != real_labels)] = 0
            pred_id.append(atk_succes)
            correct += (atk_succes*z).sum().item()

            data_stream.set_description((
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                ' => '
                'loss: {total_loss:.7f} / '
            ).format(
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss= loss,
            ))

        acc = correct 
        # /len(data_loader.dataset)
        running_loss /= len(data_loader)

    print('\nTest set: Loss: %.5f, Accuracy: %.2f %%' % (running_loss, 100. * acc))
    return acc, torch.cat(pred_id)

def process_data(vae, model, dataset, batch_size, latent_dim, data_config, output_dir, eps, n_seeds, local_op, cuda):

    x_seeds, y_seeds, op_op = torch.load('test_seeds/op_'+ model.label +'_seed.pt',torch.device('cpu'))
    adv, adv_label = torch.load('test_cases/hda/{label}.pt'.format(label = model.label),torch.device('cpu'))

    # filter the aes
    adv_eval = [ae[0] for ae in adv]
    adv_eval = torch.stack(adv_eval)
    adv_label_eval = [ae_label[0] for ae_label in adv_label]
    adv_label_eval = torch.stack(adv_label_eval)


    adv_dataset = data_utils.TensorDataset(adv_eval, adv_label_eval, op_op)
    adv_data_loader = utils.get_data_loader(adv_dataset, batch_size, cuda=cuda)

    # model trained fine tune without aes
    acc, idx= test(adv_data_loader, model, cuda)
    mask = torch.where(idx==0)

    x_seeds, y_seeds, op_op = x_seeds[mask], y_seeds[mask], op_op[mask]
    mask = mask[0].cpu()
    print(len(mask))
    adv = [adv[i] for i in mask]
    adv_label = [adv_label[i] for i in mask]

    indices = torch.randperm(len(x_seeds))
    x_seeds_rand, y_seeds_rand, op_rand = x_seeds[indices], y_seeds[indices], op_op[indices]

    # save to file
    torch.save([x_seeds_rand,y_seeds_rand,op_rand],'test_seeds/rand_'+ model.label +'_seed.pt')
    torch.save([x_seeds,y_seeds,op_op],'test_seeds/op_'+ model.label +'_seed.pt')
    torch.save([adv, adv_label],'test_cases/hda/'+ model.label +'.pt')




def robustness_eval(vae, model, dataset, batch_size, latent_dim, data_config, output_dir, eps, n_seeds, local_op, cuda):

    _, _, op_op = torch.load('RQ4/test_seeds/op_'+ model.label +'_seed.pt',torch.device('cpu'))
    

    adv, adv_label = torch.load('RQ4/evaluation/{label}_{mode}.pt'.format(label = model.label,mode='hda'),torch.device('cpu'))
    op_op = op_op/torch.sum(op_op)
    adv_dataset = data_utils.TensorDataset(adv, adv_label, op_op)
    adv_data_loader = utils.get_data_loader(adv_dataset, batch_size, cuda=cuda)

    # model trained fine tune without aes
    acc, idx= test(adv_data_loader, model, cuda)
    

    # model trained with pgd AEs
    utils.load_checkpoint_adv(model, './adv_train_checkpoints','attack')
    acc, _ = test(adv_data_loader, model, cuda)

    # model trained with coverage-guided AEs
    utils.load_checkpoint_adv(model, './adv_train_checkpoints','cov')
    acc, _ = test(adv_data_loader, model, cuda)

    # model trained with hda AEs
    utils.load_checkpoint_adv(model, './adv_train_checkpoints','hda')
    acc, _ = test(adv_data_loader, model, cuda)




    # torch.manual_seed(0)
    # vae.eval()
    # model.eval()


    # n_channel  = data_config['channels']
    # img_size = data_config['size']
    # n_class = data_config['classes']

    # n = len(dataset)
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
    #     x_mu = torch.zeros(n, latent_dim)
    #     x_std = torch.zeros(n, latent_dim)


    
    # for idx, (data, target) in enumerate(data_loader):
    #     if cuda:
    #         data, target = data.float().cuda(), target.long().cuda()
    #     else:
    #         data, target = data.float(), target.long()

    #     if len(target.size()) > 1:
    #         target = torch.argmax(target, dim=1)

    #     grad_batch = cal_gradient(model,data,target)
    #     grad_norm.append(grad_batch)

    #     with torch.no_grad():
    #         mu, log_var = vae.encode(data)
    #         hidden_act = model.hidden_act(data)
    #         target_pred = torch.argmax(model(data/2+0.5), dim=1)

    #         x_test[(idx * batch_size):((idx + 1) * batch_size), :, :, :] = data
    #         y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
    #         y_pred[(idx * batch_size):((idx + 1) * batch_size)] = target_pred
    #         x_mu[(idx * batch_size):((idx + 1) * batch_size), :] = mu
    #         x_std[(idx * batch_size):((idx + 1) * batch_size), :] = torch.exp(0.5 * log_var)
    #         x_act_dist.append(hidden_act)

    # grad_norm = torch.cat(grad_norm, dim=0)
    # x_act_dist = torch.cat(x_act_dist, dim=0)

    # indices = torch.where(y_pred==y_test)
    # x_test = x_test[indices] 
    # y_test = y_test[indices]
    # x_mu = x_mu[indices]
    # x_std = x_std[indices]
    # grad_norm = grad_norm[indices]
    # x_act_dist = x_act_dist[indices]


    # # # # ################################################################################################### 
    # start = time.time()     
    # print()
    # print('Start to test the model!')
    # print()
    # print('Dataset:', model.label)
    # print('Total No. in train set:', len(x_test))
    # print('Norm ball radius:', eps)

    # kde = GaussianKDE(x_mu, x_std)
    # pd = kde.score_samples(x_mu)
    # pd = pd/sum(pd)
    # pd =pd.cuda()

    # x_test = x_test /2 + 0.5

    # x_test = torch.split(x_test,100)
    # y_test_a = torch.split(y_test,100)


    # org_pred = []
    # attack_pred = []
    # hda_pred = []

    # for i, (x_seed,y_seed) in enumerate(zip(x_test,y_test_a)):
    #     new_input = atk(x_seed, y_seed)
    #     pred_label = torch.argmax(model(new_input), dim = 1) 
    #     org_pred.append(pred_label)

    #     pred_label = torch.argmax(model_a(new_input), dim = 1) 
    #     attack_pred.append(pred_label)

    #     pred_label = torch.argmax(model_hda(new_input), dim = 1) 
    #     hda_pred.append(pred_label)

    # org_pred = torch.cat(org_pred, dim=0)
    # attack_pred = torch.cat(attack_pred, dim=0)
    # hda_pred = torch.cat(hda_pred, dim=0)

    # mask = torch.where(org_pred!=y_test)
    # y_adv = y_test[mask]
    # pd_adv = pd[mask]

    # atk_succes = torch.zeros(len(y_adv)).cuda()
    # atk_succes[torch.where(org_pred[mask]==y_adv)] = 1
    # print('org. model R_g',torch.sum(atk_succes*pd_adv).item())

    # atk_succes = torch.zeros(len(y_adv)).cuda()
    # atk_succes[torch.where(attack_pred[mask]==y_adv)] = 1
    # print('attack model R_g',torch.sum(atk_succes*pd_adv).item())

    # atk_succes = torch.zeros(len(y_adv)).cuda()
    # atk_succes[torch.where(hda_pred[mask]==y_adv)] = 1
    # print('hda model R_g',torch.sum(atk_succes*pd_adv).item())






    

    # atk = torchattacks.PGD(model, eps=eps*0.1, alpha=2/255, steps=20)

    # new_pred = []
    # org_pred = []
    # for i, (x_seed,y_seed) in enumerate(zip(x_test,y_test_a)):
    #     new_input = atk(x_seed, y_seed)
    #     y_pred = model(new_input)
    #     pred_label = torch.argmax(y_pred, dim = 1)
    #     new_pred.append(pred_label)
    #     y_seed = torch.argmax(model(x_seed), dim=1)
    #     org_pred.append(y_seed)

    # new_pred = torch.cat(new_pred, dim=0)
    # org_pred = torch.cat(org_pred, dim=0)

    # atk_succes = torch.zeros(len(new_pred)).cuda()
    # atk_succes[torch.where((new_pred!=y_test))] = 1

    # print('org model R_g',torch.sum(atk_succes*pd).item())


    # # model trained with pgd AEs
    # utils.load_checkpoint_adv(model, './adv_train_checkpoints','attack')
    # model.eval()
    # atk = torchattacks.PGD(model, eps=eps*0.1, alpha=2/255, steps=20)

    # new_pred = []
    # org_pred = []
    # for i, (x_seed,y_seed) in enumerate(zip(x_test,y_test_a)):
    #     new_input = atk(x_seed, y_seed)
    #     y_pred = model(new_input)
    #     pred_label = torch.argmax(y_pred, dim = 1)
    #     new_pred.append(pred_label)
    #     y_seed = torch.argmax(model(x_seed), dim=1)
    #     org_pred.append(y_seed)

    # new_pred = torch.cat(new_pred, dim=0)
    # org_pred = torch.cat(org_pred, dim=0)

    # atk_succes = torch.zeros(len(new_pred)).cuda()
    # atk_succes[torch.where((new_pred!=y_test))] = 1

    # print('attack model R_g',torch.sum(atk_succes*pd).item())

    # # new_pred = []
    # # # model trained with coverage-guided AEs
    # # utils.load_checkpoint_adv(model, './adv_train_checkpoints','cov')
    # # model.eval()

    # # for i, (x_seed,y_seed) in enumerate(zip(x_test,y_test_a)):
    # #     new_input = atk(x_seed, y_seed)
    # #     y_pred = model(new_input)
    # #     pred_label = torch.argmax(y_pred, dim = 1)
    # #     new_pred.append(pred_label)

    # # new_pred = torch.cat(new_pred, dim=0)

    # # atk_succes = torch.zeros(len(new_pred)).cuda()
    # # atk_succes[torch.where(new_pred!=y_test)] = 1

    # # print('cov model R_g',torch.sum(atk_succes*pd).item())

    
    # # model trained with hda AEs
    # utils.load_checkpoint_adv(model, './adv_train_checkpoints','hda')
    # model.eval()
    # atk = torchattacks.PGD(model, eps=eps*0.1, alpha=2/255, steps=20)

    # new_pred = []
    # org_pred = []
    # for i, (x_seed,y_seed) in enumerate(zip(x_test,y_test_a)):
    #     new_input = atk(x_seed, y_seed)
    #     y_pred = model(new_input)
    #     pred_label = torch.argmax(y_pred, dim = 1)
    #     new_pred.append(pred_label)
    #     y_seed = torch.argmax(model(x_seed), dim=1)
    #     org_pred.append(y_seed)

    # new_pred = torch.cat(new_pred, dim=0)
    # org_pred = torch.cat(org_pred, dim=0)

    # atk_succes = torch.zeros(len(new_pred)).cuda()
    # atk_succes[torch.where((new_pred!=y_test))] = 1

    # print('hda model R_g',torch.sum(atk_succes*pd).item())

