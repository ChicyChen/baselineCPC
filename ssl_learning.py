import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data_UCF101 import *
from model import *

from utils import *
from augmentation import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils


"python ssl_learning.py"


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1, type=int,
                    help='model type, 0 for (1layer, 1d, static); \
                        1 for (1layer, 2d ConvGRU, static); \
                        2 for (2layer, 2d ConvGRU, static B1); \
                        3 for (2layer, 2d ConvGRU, static B2);')
parser.add_argument('--dataset', default='ucf240', type=str,
                    help='dataset name')
parser.add_argument('--which_split', default=0, type=int,
                    help='split index, 0 means using full dataset')

parser.add_argument('--num_seq', default=15, type=int,
                    help='number of frames in a seq')
parser.add_argument('--downsample', default=3, type=int)
parser.add_argument('--pred_step', default=5, type=int)
parser.add_argument('--nsub', default=5, type=int)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')

parser.add_argument('--pretrain', default='', type=str,
                    help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--prefix', default='checkpoints', type=str,
                    help='prefix of checkpoint filename')
parser.add_argument('--no_val', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--usehidden', action='store_true')
parser.add_argument('--seeall', action='store_true')


def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()
    args.useout = (not args.usehidden)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')
    
    if args.model == 0:
        model = CPC_1layer_1d_static(pred_step=args.pred_step, nsub=args.nsub)
    elif args.model == 1:
        model = CPC_1layer_2d_static(pred_step=args.pred_step, nsub=args.nsub, useout=args.useout)
    elif args.model == 2:
        model = CPC_2layer_2d_static_B1(pred_step=args.pred_step, nsub=args.nsub, useout=args.useout, seeall=args.seeall)
    elif args.model == 3:
        model = CPC_2layer_2d_static_B2(pred_step=args.pred_step, nsub=args.nsub, useout=args.useout, seeall=args.seeall)

    model = nn.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(model)
    model = model.to(cuda)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            model.load_state_dict(torch.load(args.pretrain))
            print("model loaded.")
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    global criterion
    criterion = nn.CrossEntropyLoss()
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # load data
    if args.dataset == 'ucf':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=128, consistent=True),
            Scale(size=(128,128)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
        train_loader = get_data_ucf(transform, 'train', args.num_seq, args.downsample, args.which_split, return_label=False, batch_size=args.batch_size, dim=150)
        if not args.no_val:
            test_loader = get_data_ucf(transform, 'val', args.num_seq, args.downsample, args.which_split, return_label=False, batch_size=args.batch_size, dim=150)
    if args.dataset == 'ucf240':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(224,224)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
        train_loader = get_data_ucf(transform, 'train', args.num_seq, args.downsample, args.which_split, return_label=False, batch_size=args.batch_size, dim=240)
        if not args.no_val:
            test_loader = get_data_ucf(transform, 'val', args.num_seq, args.downsample, args.which_split, return_label=False, batch_size=args.batch_size, dim=240)

    # create folders
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)

    if args.model == 0:
        model_name = '1layer_1d_static'
    elif args.model == 1:
        model_name = '1layer_2dGRU_static'
    elif args.model == 2:
        model_name = '2layer_2dGRU_static_B1'
    elif args.model == 3:
        model_name = '2layer_2dGRU_static_B2'

    ckpt_folder = os.path.join(
        args.prefix, '%s_split%s_%s_uo%s_sa%s_lr%s_wd%s_bs%s' % (args.dataset, args.which_split, model_name, args.useout, args.seeall, args.lr, args.wd, args.batch_size)) 

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    train_loss_list = []
    test_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    # start training
    for epoch in epoch_list:
        train_loss = train(
            train_loader, model, optimizer, epoch)
        train_loss_list.append(train_loss)

        if not args.no_val:
            test_loss = train(
                test_loader, model, optimizer, epoch, False)
            test_loss_list.append(test_loss)
            if test_loss < lowest_loss:
                lowest_loss = test_loss
                best_epoch = epoch + 1
        else:
            if train_loss < lowest_loss:
                lowest_loss = train_loss
                best_epoch = epoch + 1

        # save models
        if not args.no_save:
            checkpoint_path = os.path.join(
                ckpt_folder, 'epoch%s.pth.tar' % str(epoch+1))
            torch.save(model.state_dict(), checkpoint_path)

    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))
    print('Best epoch: %s' % best_epoch)


    # plot training process
    plt.plot(epoch_list, train_loss_list, label = 'train')
    if not args.no_val:
        plt.plot(epoch_list, test_loss_list, label = 'val')
    plt.title('Train and test loss')
    plt.xticks(epoch_list, epoch_list)
    plt.legend()
    plt.savefig(os.path.join(
        ckpt_folder, 'epoch%s_bs%s.png' % (epoch+1, args.batch_size)))


"""
def train(data_loader, model, optimizer, epoch, train = True):
    if train:
        model.train()
    else:
        model.eval()

    loss_list = []

    for _, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        [score, mask] = model(input_seq)
        loss = criterion(score, mask)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            pass
        loss_list.append(loss.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    if train:
        print('Epoch:', epoch, '; Train loss:', mean_loss)
    else:
        print('Epoch:', epoch, '; Validation loss:', mean_loss)

    return mean_loss
"""

def train(data_loader, model, optimizer, epoch, train = True):
    if train:
        model.train()
    else:
        model.eval()

    loss_list = []

    for _, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        loss = model(input_seq)
        if train:
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        else:
            pass
        loss_list.append(loss.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    if train:
        print('Epoch:', epoch, '; Train loss:', mean_loss)
    else:
        print('Epoch:', epoch, '; Validation loss:', mean_loss)

    return mean_loss



if __name__ == '__main__':
    main()
