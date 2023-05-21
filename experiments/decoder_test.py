import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
# from tensorboardX import SummaryWriter

from data_MNIST import *
from utils.utils import *
from baseline import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from PIL import Image

"python decoder_test.py"
"python decoder_test.py --epoch 3"
"python decoder_test.py --epoch 10 --gpu 2"
"python decoder_test.py --epoch 10 --gpu 3 --la 0.1"


parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--nsub', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--la', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--prefix', default='checkpoints', type=str,
                    help='prefix of checkpoint filename')
parser.add_argument('--visual', default='visualize', type=str,
                    help='visualize folder name')


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    global vis
    vis = 0

    model = CPC_1layer_1d_static_decoder(pred_step=args.pred_step, nsub=args.nsub)

    checkpoint_path = os.path.join(
        args.prefix, 'CPC_1layer_1d_static_decoder_lr%s_wd%s_la%s_bs%s' % (args.lr, args.wd, args.la, args.batch_size), 'epoch%s.pth.tar' % str(args.epoch)) 
    model.load_state_dict(torch.load(checkpoint_path))
    # print(model.state_dict())

    # model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion, mse
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.], [1.])
    ])

    test_loader = get_data(transform, 'test', args.num_seq, args.downsample, return_motion=False, return_digit=False, batch_size=args.batch_size)

    # create folders
    if not os.path.exists(args.visual):
        os.makedirs(args.visual)

    vis_folder = os.path.join(
        args.visual, 'CPC_1layer_1d_static_decoder_lr%s_wd%s_la%s_epoch%s' % (args.lr, args.wd, args.la, args.epoch))
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    test_loss, test_mse = test(test_loader, model, vis_folder, args.la)

    print('Testing finished')


def test(data_loader, model, vis_folder, la):
    global vis
    loss_list = []
    mse_list = []
    model.eval()
    for idx, input_seq in enumerate(data_loader):
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_, reconst_] = model(input_seq)
        # print(score_.shape)
        (B, pred, B, N) = score_.shape
        # print(B, pred, N)
        score_flattened = score_.view(B*pred, B*N)
        mask_flattened = mask_.view(B*pred, B*N)
        mask_flattened = mask_flattened.to(int).argmax(dim=1)
        # mask_flattened = mask_flattened.to(cuda)
        loss = criterion(score_flattened, mask_flattened)
        loss_mse = mse(input_seq[:, -pred:, :, :, :], reconst_)
        total_loss = la*loss + (1-la)*loss_mse
        loss_list.append(total_loss.cpu().detach().numpy())
        mse_list.append(loss_mse.cpu().detach().numpy())
        if idx % 10 == 0:
            filename = os.path.join(vis_folder, '%d.png' % vis)
            visualize_pred(
                input_seq[0, -pred:, :, :, :].cpu().detach().numpy(), reconst_[0].cpu().detach().numpy(), filename)
            vis = vis + 1

    mean_loss = np.mean(loss_list)
    mean_mse = np.mean(mse_list)
    print('Loss:', mean_loss, '; MSE:', mean_mse)

    return mean_loss, mean_mse



if __name__ == '__main__':
    main()
