import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data_MNIST import *
from utils.utils import *
from hierarchical import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

"python hierarchical_train.py"
"python hierarchical_train.py --no_save"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--nsub', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--pretrain', default='', type=str,
                    help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--prefix', default='checkpoints', type=str,
                    help='prefix of checkpoint filename')
parser.add_argument('--no_test', action='store_true')
parser.add_argument('--no_save', action='store_true')


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    model = CPC_2layer_1d_static(pred_step=args.pred_step, nsub=args.nsub)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            model.load_state_dict(torch.load(args.pretrain))
            print("model loaded.")
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    # model = nn.DataParallel(model)
    model = model.to(cuda)

    global criterion
    criterion = nn.CrossEntropyLoss()
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.], [1.])
    ])

    train_loader = get_data(transform, 'train', args.num_seq, args.downsample, return_motion=False, return_digit=False, batch_size=args.batch_size)
    if not args.no_test:
        test_loader = get_data(transform, 'test', args.num_seq, args.downsample, return_motion=False, return_digit=False, batch_size=args.batch_size)

    # create folders
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)

    ckpt_folder = os.path.join(
        args.prefix, 'CPC_2layer_1d_static_lr%s_wd%s_bs%s' % (args.lr, args.wd, args.batch_size)) 
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    train_loss_list = []
    test_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    for epoch in epoch_list:
        train_loss = train(
            train_loader, model, optimizer, epoch)
        train_loss_list.append(train_loss)

        if not args.no_test:
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
    if not args.no_test:
        plt.plot(epoch_list, test_loss_list, label = 'test')
    plt.title('Train and test loss')
    plt.xticks(epoch_list, epoch_list)
    plt.legend()
    plt.savefig(os.path.join(
        ckpt_folder, 'epoch%s_bs%s.png' % (epoch+1, args.batch_size)))



def train(data_loader, model, optimizer, epoch, train = True):
    if train:
        model.train()
    else:
        model.eval()

    loss_list = []

    for idx, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, score2_, mask_] = model(input_seq)
        (B, pred, B, N) = score_.shape
        score_flattened = score_.view(B*pred, B*N)
        score2_flattened = score2_.view(B*pred, B*N)
        mask_flattened = mask_.view(B*pred, B*N)
        mask_flattened = mask_flattened.to(int).argmax(dim=1)
        loss = criterion(score_flattened, mask_flattened) + criterion(score2_flattened, mask_flattened)
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
        print('Epoch:', epoch, '; Test loss:', mean_loss)

    return mean_loss



if __name__ == '__main__':
    main()
