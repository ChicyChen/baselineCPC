import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
# from tensorboardX import SummaryWriter

from baseline_data import *
from baseline_cpc_with_decoder import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from PIL import Image

"python baseline_test_with_decoder.py"
"python baseline_test_with_decoder.py --epoch 20"
"python baseline_test_with_decoder.py --la 0.1 --epoch 20"

parser = argparse.ArgumentParser()
parser.add_argument('--num_seq', default=10, type=int,
                    help='number of video blocks')
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--nsub', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--la', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--prefix', default='wd_checkpoint', type=str,
                    help='prefix of checkpoint filename')


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

    model = baseline_CPC_with_decoder(pred_step=args.pred_step, nsub=args.nsub)

    checkpoint_path = os.path.join(
        args.prefix, 'epoch%s.pth.tar' % str(args.epoch))
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

    test_loader = get_data(transform, 'test')

    test_loss = test(test_loader, model, args.la)

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


def test(data_loader, model, la=0.5):
    global vis
    loss_list = []
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
        if idx % 10 == 0:
            visualize_pred(
                input_seq[0, -pred:, :, :, :].cpu().detach().numpy(), reconst_[0].cpu().detach().numpy())

    mean_loss = np.mean(loss_list)
    print('Loss:', mean_loss)

    return mean_loss


def visualize_pred(gt_frames, pred_frames):
    global vis
    pred, C, H, W = gt_frames.shape
    swap_gt = np.swapaxes(gt_frames, 0, 1)
    swap_gt = swap_gt.reshape(C, pred*H, W).transpose(1, 2, 0)
    swap_pred = np.swapaxes(pred_frames, 0, 1)
    swap_pred = swap_pred.reshape(C, pred*H, W).transpose(1, 2, 0)
    full_img = np.concatenate((swap_gt, swap_pred), axis=1)
    full_img = full_img*(255/np.max(full_img))
    im = Image.fromarray(full_img.astype(np.uint8))
    im.save("fun_vis_{}.jpg".format(vis))
    vis = vis + 1


if __name__ == '__main__':
    main()
