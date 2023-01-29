import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from data_MNIST import *
from utils import *
from baseline import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

"python baseline_test.py"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--nsub', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--prefix', default='checkpoints', type=str,
                    help='prefix of checkpoint filename')


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    model = CPC_1layer_1d_static(pred_step=args.pred_step, nsub=args.nsub)

    checkpoint_path = os.path.join(
        args.prefix, 'CPC_1layer_1d_static_lr%s_wd%s_bs%s' % (args.lr, args.wd, args.batch_size), 'epoch%s.pth.tar' % str(args.epoch))
    model.load_state_dict(torch.load(checkpoint_path))
    # print(model.state_dict())

    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.], [1.])
    ])

    test_loader = get_data(transform, 'test', args.num_seq, args.downsample, return_motion=False, return_digit=False, batch_size=args.batch_size)

    test_loss = test(
        test_loader, model, criterion)

    print('Testing finished')


def test(data_loader, model, criterion):
    loss_list = []
    model.eval()
    for idx, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)
        # print(score_.shape)
        (B, pred, B, N) = score_.shape
        # print(B, pred, N)
        score_flattened = score_.view(B*pred, B*N)
        mask_flattened = mask_.view(B*pred, B*N)
        mask_flattened = mask_flattened.to(int).argmax(dim=1)
        # mask_flattened = mask_flattened.to(cuda)
        loss = criterion(score_flattened, mask_flattened)
        loss_list.append(loss.cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    print('Loss:', mean_loss)

    return mean_loss


if __name__ == '__main__':
    main()
