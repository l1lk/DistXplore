from torch import optim
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.utils.data as data_utils
from tqdm import tqdm
import utils

def train(data_loader, model, optimizer, epoch, cuda):
    model.train()
    running_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    for batch_index, (x, y) in data_stream:

        # prepare data on gpu if needed
        if cuda:
            x = x.to('cuda')
            y = y.to('cuda')

        x = x/2 + 0.5
        
        if len(y.size()) > 1:
            real_labels = torch.argmax(y, dim=1)
        else:
            real_labels = y
        
        optimizer.zero_grad()
        result = model(x)
        loss = criterion(result, real_labels)

        # backprop gradients from the loss
        loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(result, dim=1)
        correct += (pred_labels == real_labels).sum().item() 
        running_loss += loss.item()

        # update progress
        data_stream.set_description((
            'epoch: {epoch} | '
            'progress: [{trained}/{total}] ({progress:.0f}%) | '
            ' => '
            'loss: {loss:.7f} / '
        ).format(
            epoch=epoch,
            trained=batch_index * len(x),
            total=len(data_loader.dataset),
            progress=(100. * batch_index / len(data_loader)),
            loss = loss.data.item()
        ))
    
    acc = correct / len(data_loader.dataset)
    running_loss /= len(data_loader)
    print('\nTraining set: Epoch: %d, Loss: %.5f, Accuracy: %.2f %%' % (epoch, running_loss, 100. * acc))
    return acc


def test(data_loader, model,cuda):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        for batch_index, (x, y) in data_stream:
            # prepare data on gpu if needed
            if cuda:
                x = x.to('cuda')
                y = y.to('cuda')
            
            x = x/2 + 0.5

            if len(y.size()) > 1:
                real_labels = torch.argmax(y, dim=1)
            else:
                real_labels = y

            result = model(x)
            loss = criterion(result, real_labels)
            running_loss += loss.item()

            pred_labels = torch.argmax(result, dim=1)
            correct += (pred_labels == real_labels).sum().item()

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

        acc = correct / len(data_loader.dataset)
        running_loss /= len(data_loader)

    print('\nTest set: Loss: %.5f, Accuracy: %.2f %%' % (running_loss, 100. * acc))
    return acc



def adv_train_model(model, train_dataset, test_dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                checkpoint_dir='./checkpoints',
                mode = 'hda',
                rou = 1,
                cuda=True):

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=5)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])

    # scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
    # scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])
   

    adv, adv_label = torch.load('RQ4/test_cases/{label}_{mode}.pt'.format(label = model.label,mode=mode),torch.device('cpu'))


    _, _, op_rand = torch.load('RQ4/test_seeds/rand_'+ model.label +'_seed.pt',torch.device('cpu'))
    _, _, op_op = torch.load('RQ4/test_seeds/op_'+ model.label +'_seed.pt')

    op_rand = op_rand / torch.sum(op_rand)
    op_op = op_op / torch.sum(op_op)

    n = int(len(adv)*rou)
    adv = adv[:n]
    adv_label = adv_label[:n]

    adv = torch.cat(adv)
    adv_label = torch.cat(adv_label)
    adv = (adv-0.5)*2

    print('random seeds density:',torch.sum(op_rand[:n]).item())
    print('op seeds density:',torch.sum(op_op[:n]).item())


    epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    epoch_start = 1

    org_train_loader = utils.get_data_loader(train_dataset, batch_size, cuda=cuda)

    adv_dataset = data_utils.TensorDataset(adv, adv_label)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset,adv_dataset])

    train_data_loader = utils.get_data_loader(train_dataset, batch_size, cuda=cuda)
    test_data_loader = utils.get_data_loader(test_dataset, batch_size, cuda=cuda)

    BEST_acc = 0.0
    LAST_SAVED = -1

    for epoch in range(epoch_start, epochs+1):
        
        train(train_data_loader, model, optimizer, epoch, cuda)
        # acc = test(test_data_loader, model, cuda)
        scheduler.step()
            
        # print()
        # if acc >= BEST_acc:
        #     BEST_acc = acc
        #     LAST_SAVED = epoch
        #     print("Saving model!")
        #     utils.save_checkpoint_adv(model, mode, './adv_train_checkpoints', epoch)
        # else:
        #     print("Not saving model! Last saved: {}".format(LAST_SAVED))
    acc = test(org_train_loader, model, cuda)
    acc = test(test_data_loader, model, cuda)
    print("Saving model!")
    utils.save_checkpoint_adv(model, mode, './adv_train_checkpoints', epoch)


    # adv_train_checkpoints
    # checkpoints_fine_tune