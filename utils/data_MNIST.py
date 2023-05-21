import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
import glob
import csv
import pandas as pd

import numpy as np
from PIL import Image

from utils import *
from augmentation import *


def get_data(transform=None, mode='train', num_seq=10, downsample=2, return_motion=False, return_digit=False, batch_size=2):
    print('Loading data for "%s" ...' % mode)
    dataset = Moving_MNIST(mode=mode,
                           transform=transform,
                           num_seq=num_seq,
                           downsample=downsample,
                           return_motion=return_motion,
                           return_digit=return_digit)
    sampler = data.RandomSampler(dataset)
    if mode == 'train' or mode == 'train_ft':
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      #   num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      #   num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


class Moving_MNIST(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 num_seq=8,
                 downsample=3,
                 return_motion=False,
                 return_digit=False):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_motion = return_motion
        self.return_digit = return_digit

        print('Using Moving MNIST data (64x64)')

        # get motion list
        motion_list = ["vertical", "horizontal", "circular_clockwise",
                       "circular_anticlockwise", "zigzag", "tofro"]
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        for id in range(len(motion_list)):
            self.action_dict_decode[id] = motion_list[id]
            self.action_dict_encode[motion_list[id]] = id

        # splits
        if mode == 'train':
            split = 'data/movingmnist/train.csv'
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test':
            split = 'data/movingmnist/test.csv'
            video_info = pd.read_csv(split, header=None)
        elif mode == 'train_ft':
            split = 'data/movingmnist/train_ft.csv'
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test_ft':
            split = 'data/movingmnist/test_ft.csv'
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')
        self.video_info = video_info

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.downsample <= 0:
            return [None]
        n = 1
        start_idx = np.random.choice(
            range(vlen-self.num_seq*self.downsample), n)
        seq_idx = np.arange(self.num_seq)*self.downsample + start_idx
        return [seq_idx, vpath]

    def __getitem__(self, index):
        vlen = 30
        vpath, digit, motion_type = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        # print(idx_block.shape)
        # print(self.num_seq)
        assert idx_block.shape == (self.num_seq,)

        # print(os.path.join(vpath, '{}.jpg'.format(0)))

        t_seq = [pil_loader(os.path.join(vpath, '{}.jpg'.format(i)))
                 for i in idx_block]
        C = 3
        (H, W) = t_seq[0].size
        t_seq = [self.transform(frame)
                 for frame in t_seq]  # apply same transform
        t_seq = torch.stack(t_seq)
        t_seq = t_seq.view(self.num_seq, C, H, W)

        if self.return_motion and self.return_digit:
            motion = torch.LongTensor([self.encode_action(motion_type)])
            digit = torch.LongTensor([digit])
            return t_seq, motion, digit
        if self.return_motion and not self.return_digit:
            # print(motion_type)
            motion = torch.LongTensor([self.encode_action(motion_type)])
            return t_seq, motion
        if not self.return_motion and self.return_digit:
            digit = torch.LongTensor([digit])
            return t_seq, digit
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
