import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data import *
from utils import *
from bcpc_model import *
from blc_model import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F


"python blcm_test.py"
"python blcm_test.py --epoch 2"



parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('--backbone_folder', default='checkpoints/bcpc_lr0.0001_wd1e-05', type=str,
                    help='path of pretrained backbone or model')
parser.add_argument('--backbone_epoch', default=5, type=int,
                    help='epoch of pretrained backbone or model')

parser.add_argument('--epoch', default=10, type=int,
                    help='number of total epochs to finetune')
parser.add_argument('--gpu', default='1', type=str)



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    model = baseline_m_lc()
    ckpt_folder = os.path.join(
        args.backbone_folder, 'ftmotion_lr%s_wd%s_ep%s' % (args.lr, args.wd, args.backbone_epoch))
    checkpoint_path = os.path.join(ckpt_folder, 'epoch%s.pth.tar' % args.epoch)
    model.load_state_dict(torch.load(checkpoint_path))

    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion
    criterion = nn.CrossEntropyLoss()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.], [1.])
    ])

    test_loader = get_data(transform, 'test_ft', args.num_seq, args.downsample, return_motion=True, return_digit=False, batch_size=args.batch_size)
        
    test_acc = test(
        test_loader, model, criterion)
            
    print('Testing finished')
   


def test(data_loader, model, criterion):
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

        loss_list.append(loss.cpu().detach().numpy())
        acc_list.append(acc.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)
    
    print('Test loss:', mean_loss, 'Test Acc:', mean_acc)

    return mean_acc


if __name__ == '__main__':
    main()
