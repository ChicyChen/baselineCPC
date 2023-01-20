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

"python baseline_test.py"
"python baseline_test.py --epoch 20"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--nsub', default=3, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--prefix', default='checkpoint', type=str,
                    help='prefix of checkpoint filename')


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    model = baseline_CPC(pred_step=args.pred_step, nsub=args.nsub)

    checkpoint_path = os.path.join(
        args.prefix, 'epoch%s.pth.tar' % str(args.epoch))
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

    test_loader = get_data(transform, 'test')

    test_loss = test(
        test_loader, model, criterion)

    print('Testing finished')


def get_data(transform=None, mode='test'):
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
