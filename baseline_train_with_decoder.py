import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from baseline_data import *
from baseline_cpc_with_decoder import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
# parser.add_argument('--resume', default='', type=str,
#                     help='path of model to resume')
# TODO pre-training
# parser.add_argument('--pretrain', default='', type=str,
#                     help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int,
                    help='frequency of printing output during training')
parser.add_argument('--prefix', default='wd_checkpoint', type=str,
                    help='prefix of checkpoint filename')


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    model = baseline_CPC_with_decoder(pred_step=args.pred_step)
    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion, mse
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    lowest_loss = np.inf
    global iteration
    iteration = 0

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
    checkpoint_path = os.path.join(args.prefix, 'epoch%s.pth.tar' % str(epoch+1))
    torch.save(model.state_dict(), checkpoint_path)


def get_data(transform=None, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = Moving_MNIST(mode=mode,
                           transform=transform,
                           num_seq=args.num_seq,
                           downsample=args.downsample)
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
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_, reconst_] = model(input_seq) # reconst_: (B, pred_step, C, H, W)
        # print(score_.shape)
        (B, pred, B, N) = score_.shape
        # print(B, pred, N)
        score_flattened = score_.view(B*pred, B*N)
        mask_flattened = mask_.view(B*pred, B*N)
        mask_flattened = mask_flattened.to(int).argmax(dim=1)
        # mask_flattened = mask_flattened.to(cuda)
        loss = criterion(score_flattened, mask_flattened)
        loss_mse = mse(input_seq[:, -pred:, :, :, :], reconst_)
        optimizer.zero_grad()
        total_loss = 0.5*loss + 0.5*loss_mse
        total_loss.backward()
        optimizer.step()
        loss_list.append(total_loss.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    print('Epoch:', epoch, 'Loss:', mean_loss)

    return mean_loss




if __name__ == '__main__':
    main()
