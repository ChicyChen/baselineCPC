import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from baseline_data import *
from baseline_cpc import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F

"python classify_motion.py"
"python classify_motion.py --backbone noexit --prefix checkpoint_m_lc_noprecl"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
# parser.add_argument('--resume', default='', type=str,
#                     help='path of model to resume')
parser.add_argument('--backbone', default='checkpoint/epoch10.pth.tar', type=str,
                    help='path of pretrained backbone')

# parser.add_argument('--pretrain', default='', type=str,
#                     help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--prefix', default='checkpoint_m_lc', type=str,
                    help='prefix of checkpoint filename')


class baseline_m_lc(nn.Module):
    # pre_frame_num = 5, pred_step = 3, in total 8 frames
    def __init__(self, code_size=128):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.mask = None

        self.genc = nn.Sequential(
            # (X, 3, 64, 64) -> (X, 16, 32, 32)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (X, 16, 32, 32) -> (X, 32, 16, 16)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (X, 32, 16, 16) -> (X, 64, 8, 8)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 8, 8) -> (X, 64, 4, 4)
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.code_size)
        )

        self.gar = nn.GRU(self.code_size, 256, batch_first=True)

        self.pred = nn.Linear(256, self.code_size)

        self.predmotion = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)
        # self._initialize_weights(self.predmotion)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        context, _ = self.gar(feature.contiguous())
        context = context[:, -1, :]
        # print(context.size())
        output = self.predmotion(context).view(B, 6)
        # print(output.size())

        return [output, context]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def get_data(transform=None, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = Moving_MNIST(mode=mode,
                           transform=transform,
                           num_seq=args.num_seq,
                           downsample=args.downsample,
                           return_motion=True)
    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      #   num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      #   num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def train(data_loader, model, optimizer, epoch):
    loss_list = []
    acc_list = []
    model.train()

    for idx, (input_seq, motion_lb) in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        motion_lb = motion_lb.to(cuda)
        # print(motion_lb.size())

        B = input_seq.size(0)
        motion_lb = motion_lb.view(B,)
        # print(motion_lb.size())

        [output, _] = model(input_seq)

        loss = criterion(output, motion_lb)
        acc = calc_accuracy(output, motion_lb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy())
        acc_list.append(acc.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)
    print('Epoch:', epoch, 'Loss:', mean_loss, 'Acc:', mean_acc)

    return mean_loss


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    model = baseline_m_lc()
    if os.path.isfile(args.backbone):
        print("=> loading pretrained backbone '{}'".format(args.backbone))
        backbone = torch.load(args.backbone)
        model = neq_load_customized(model, backbone)
        print("backbone loaded.")
    else:
        print("=> no backbone found at '{}'".format(args.backbone))

    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion
    criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    params = None
    if os.path.isfile(args.backbone):
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'predmotion' in name:
                params.append({'params': param})
            else:
                params.append({'params': param, 'lr': args.lr/10})
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

    train_loader = get_data(transform, 'train')

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(
            train_loader, model, optimizer, epoch)
        lowest_loss = min(train_loss, lowest_loss)
        # print('train loss: {}'.format(train_loss))
    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))

    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    checkpoint_path = os.path.join(
        args.prefix, 'epoch%s.pth.tar' % str(epoch+1))
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
