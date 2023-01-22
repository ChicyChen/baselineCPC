import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
# from tensorboardX import SummaryWriter

from data import *
from utils import *
from bcpc_model import *

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from PIL import Image

"python baseline_test_with_decoder_after_finetune.py"
"python baseline_test_with_decoder_after_finetune.py --epoch 20"
"python baseline_test_with_decoder_after_finetune.py --la 0.1 --prefix wd_checkpoint_la01 --epoch 20"
"python baseline_test_with_decoder_after_finetune.py --la 0.1 --prefix wd_checkpoint_la01"

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
parser.add_argument('--finetuned', default='wd_checkpoint_la01_m_lc/epoch10.pth.tar', type=str,
                    help='path of finetuned model')


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

    finetuned = torch.load(args.finetuned)
    model = neq_load_customized(model, finetuned)

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

    test_loss = test(test_loader, model, args.la)

    print('Testing finished')


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


if __name__ == '__main__':
    main()