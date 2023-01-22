import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
# from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from PIL import Image

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


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def visualize_pred(gt_frames, pred_frames, filename):
    pred, C, H, W = gt_frames.shape
    swap_gt = np.swapaxes(gt_frames, 0, 1)
    swap_gt = swap_gt.reshape(C, pred*H, W).transpose(1, 2, 0)
    swap_pred = np.swapaxes(pred_frames, 0, 1)
    swap_pred = swap_pred.reshape(C, pred*H, W).transpose(1, 2, 0)
    full_img = np.concatenate((swap_gt, swap_pred), axis=1)
    full_img = full_img*(255/np.max(full_img))
    im = Image.fromarray(full_img.astype(np.uint8))
    im.save(filename)