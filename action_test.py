
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data_UCF101 import *
from data_HMDB51 import *
from model import *
from model_B import *
from model_A import *
from model_R import *
from model_M import *

from utils import *
from augmentation import *

import logging

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1, type=int,
                    help='model type')
parser.add_argument('--dataset', default='hmdb240', type=str,
                    help='dataset name')
parser.add_argument('--which_split', default=0, type=int)

parser.add_argument('--num_seq', default=8, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--batch_size', default=1, type=int)


parser.add_argument('--backbone_folder', default='checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_saFalse_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep10', type=str,
                    help='path of pretrained backbone or model')
parser.add_argument('--backbone_epoch', default=10, type=int,
                    help='epoch of pretrained backbone or model')


parser.add_argument('--gpu', default='0,1,2,3', type=str)

parser.add_argument('--lada', default=0.1, type=float, help='h parameter for model R')


def main():
    torch.manual_seed(233)
    np.random.seed(233)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    if args.model == 0:
        if args.dataset == 'ucf':
            model = action_CPC_1layer_1d_static(class_num = 101)
        if args.dataset == 'hmdb':
            model = action_CPC_1layer_1d_static(class_num = 51)
    elif args.model == 1:
        if args.dataset == 'ucf240':
            model = action_CPC_1layer_2d_static(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_1layer_2d_static(class_num = 51)
    elif args.model == 2:
        if args.dataset == 'ucf240':
            model = action_CPC_2layer_2d_static_B1(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_2layer_2d_static_B1(class_num = 51)
    elif args.model == 3:
        if args.dataset == 'ucf240':
            model = action_CPC_2layer_2d_static_B2(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_2layer_2d_static_B2(class_num = 51)
    elif args.model == 4:
        if args.dataset == 'ucf240':
            model = action_CPC_2layer_2d_static_A1(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_2layer_2d_static_A1(class_num = 51)
    elif args.model == 5:
        if args.dataset == 'ucf240':
            model = action_CPC_2layer_2d_static_A2(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_2layer_2d_static_A2(class_num = 51)
    elif args.model == 6:
        if args.dataset == 'ucf240':
            model = action_CPC_1layer_2d_static_R1(class_num = 101, lada=args.lada)
        if args.dataset == 'hmdb240':
            model = action_CPC_1layer_2d_static_R1(class_num = 51, lada=args.lada)
    elif args.model == 7:
        if args.dataset == 'ucf240':
            model = action_CPC_1layer_2d_static_M0(class_num = 101)
        if args.dataset == 'hmdb240':
            model = action_CPC_1layer_2d_static_M0(class_num = 51)

    model = nn.DataParallel(model)
    model = model.to(cuda)

    backbone_path = os.path.join(args.backbone_folder, 'epoch%s.pth.tar' % args.backbone_epoch)
    if os.path.isfile(backbone_path):
        print("=> loading pretrained backbone '{}'".format(backbone_path))
        backbone = torch.load(backbone_path)
        model = neq_load_customized(model, backbone)
        print("backbone loaded.")
    else:
        print("=> no backbone found at '{}'".format(backbone_path))
        raise ValueError('no trained model found.')

    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(224,224)),
        ToTensor(),
        Normalize()
    ])
    # transform = transforms.Compose([
    #     RandomSizedCrop(consistent=True, size=224, p=1.0),
    #     Scale(size=(224,224)),
    #     RandomHorizontalFlip(consistent=True),
    #     ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
    #     ToTensor(),
    #     Normalize()
    # ])
    if args.dataset == 'ucf240':
        test_loader = get_data_ucf(transform, 'test', args.num_seq, args.downsample, args.which_split, return_label=True, batch_size=args.batch_size, dim = 240)
    if args.dataset == 'hmdb240':
        test_loader = get_data_hmdb(transform, 'test', args.num_seq, args.downsample, args.which_split, return_label=True, batch_size=args.batch_size, dim = 240)
    
    test_file = os.path.join(args.backbone_folder, 'test_epoch%s_split%s.log' % (args.backbone_epoch, args.which_split))
    logging.basicConfig(filename=test_file, level=logging.INFO)
    logging.info('Started')

    test_acc = test(test_loader, model)

    # with open(test_file, 'w') as f:
    #     f.write('test_acc: '+str(test_acc))


def test(data_loader, model):
    model.eval()
    acc_list = []

    for idx, (input_seq, motion_lb) in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        motion_lb = motion_lb.to(cuda)
        B = input_seq.size(0)
        motion_lb = motion_lb.view(B,)

        output = model(input_seq)
        acc = calc_accuracy(output, motion_lb)
        acc_list.append(acc.cpu().detach().numpy())

    mean_acc = np.mean(acc_list)

    print('Test--', 'Accuracy:', mean_acc)
    logging.info('Test--Accuracy: %s' % mean_acc)

    return mean_acc


if __name__ == '__main__':
    main()