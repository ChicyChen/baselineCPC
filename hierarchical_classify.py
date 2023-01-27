import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data import *
from utils import *
from hierarchical import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F


"python hierarchical_classify.py --motion --backbone_folder checkpoints/CPC_2layer_1d_static_lr0.0001_wd1e-05_bs128 --gpu 0 --freeze"
"python hierarchical_classify.py --backbone_folder checkpoints/CPC_2layer_1d_static_lr0.0001_wd1e-05_bs128 --gpu 2 --freeze"
"python hierarchical_classify.py --motion --backbone_folder checkpoints/CPC_2layer_1d_static_lr0.0001_wd1e-05_bs128 --gpu 1"
"python hierarchical_classify.py --backbone_folder checkpoints/CPC_2layer_1d_static_lr0.0001_wd1e-05_bs128 --gpu 3"

"python hierarchical_classify.py --motion --backbone_folder checkpoints/CPC_2layer_1d_static_nossl --gpu 0 --freeze"
"python hierarchical_classify.py --backbone_folder checkpoints/CPC_2layer_1d_static_nossl --gpu 2 --freeze"
"python hierarchical_classify.py --motion --backbone_folder checkpoints/CPC_2layer_1d_static_nossl --gpu 1"
"python hierarchical_classify.py --backbone_folder checkpoints/CPC_2layer_1d_static_nossl --gpu 3"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('--backbone_folder', default='checkpoints/bcpc_lr0.0001_wd1e-05', type=str,
                    help='path of pretrained backbone or model')
parser.add_argument('--backbone_epoch', default=10, type=int,
                    help='epoch of pretrained backbone or model')

parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to finetune')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--no_test', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--motion', action='store_true')



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    if args.motion:
        model = motion_CPC_2layer_1d_static()
    else:
        model = digit_CPC_2layer_1d_static()

    backbone_path = os.path.join(args.backbone_folder, 'epoch%s.pth.tar' % args.backbone_epoch)
    if os.path.isfile(backbone_path):
        print("=> loading pretrained backbone '{}'".format(backbone_path))
        backbone = torch.load(backbone_path)
        model = neq_load_customized(model, backbone)
        print("backbone loaded.")
    else:
        print("=> no backbone found at '{}'".format(backbone_path))

    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion
    criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    params = None
    # if os.path.isfile(backbone_path):
    if not args.freeze:
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'predmotion' in name or 'preddigit' in name:
                params.append({'params': param})
            else:
                params.append({'params': param, 'lr': args.lr/10})
        print(len(params))
    else:
        print('=> freeze backbone')
        params = []
        for name, param in model.named_parameters():
            if 'predmotion' in name or 'preddigit' in name:
                params.append({'params': param})
            else:
                param.requires_grad = False
                params.append({'params': param})
        print(len(params))
        

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    if params is None:
        params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    lowest_loss = np.inf

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.], [1.])
    ])

    if args.motion:
        train_loader = get_data(transform, 'train_ft', args.num_seq, args.downsample, return_motion=True, return_digit=False, batch_size=args.batch_size)
        if not args.no_test:
            test_loader = get_data(transform, 'test_ft', args.num_seq, args.downsample, return_motion=True, return_digit=False, batch_size=args.batch_size)
        # create folders
        if not args.freeze:
            ckpt_folder = os.path.join(
                args.backbone_folder, 'ftmotion_lr%s_wd%s_ep%s' % (args.lr, args.wd, args.backbone_epoch)) 
        else:
            ckpt_folder = os.path.join(
                args.backbone_folder, 'freeze_ftmotion_lr%s_wd%s_ep%s' % (args.lr, args.wd, args.backbone_epoch)) 
    else:
        train_loader = get_data(transform, 'train_ft', args.num_seq, args.downsample, return_motion=False, return_digit=True, batch_size=args.batch_size)
        if not args.no_test:
            test_loader = get_data(transform, 'test_ft', args.num_seq, args.downsample, return_motion=False, return_digit=True, batch_size=args.batch_size)
        # create folders
        if not args.freeze:
            ckpt_folder = os.path.join(
                args.backbone_folder, 'ftdigit_lr%s_wd%s_ep%s' % (args.lr, args.wd, args.backbone_epoch)) 
        else:
            ckpt_folder = os.path.join(
                args.backbone_folder, 'freeze_ftdigit_lr%s_wd%s_ep%s' % (args.lr, args.wd, args.backbone_epoch))
   
    
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    train_acc_list = []
    test_acc_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_acc = train(
            train_loader, model, optimizer, epoch, train = True)
        train_acc_list.append(train_acc)

        if not args.no_test:
            test_acc = train(
                test_loader, model, optimizer, epoch, train = False)
            test_acc_list.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
        else:
            if train_acc > best_acc:
                best_acc = train_acc
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
    plt.plot(epoch_list, train_acc_list, label = 'train')
    if not args.no_test:
        plt.plot(epoch_list, test_acc_list, label = 'test')
    plt.title('Train and test acc')
    plt.xticks(epoch_list, epoch_list)
    plt.legend()
    plt.savefig(os.path.join(
        ckpt_folder, 'epoch%s_bs%s.png' % (epoch+1, args.batch_size)))



def train(data_loader, model, optimizer, epoch, train):
    if train:
        model.train()
    else:
        model.eval()

    loss_list = []
    acc_list = []

    for idx, (input_seq, motion_lb) in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        motion_lb = motion_lb.to(cuda)

        B = input_seq.size(0)
        motion_lb = motion_lb.view(B,)

        [output, _] = model(input_seq)

        loss = criterion(output, motion_lb)
        acc = calc_accuracy(output, motion_lb)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            pass

        loss_list.append(loss.cpu().detach().numpy())
        acc_list.append(acc.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)

    if train:
        print('Epoch:', epoch, 'Train loss:', mean_loss, 'Train Acc:', mean_acc)
    else:
        print('Epoch:', epoch, 'Test loss:', mean_loss, 'Test Acc:', mean_acc)

    return mean_acc


if __name__ == '__main__':
    main()
