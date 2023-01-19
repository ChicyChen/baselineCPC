# TODO generate Moving MNIST data [video, digit, moving type]

import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
import glob
import csv
import pandas as pd

import numpy as np
import cv2
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# TODO fit with generated moving MNIST
class Moving_MNIST(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=8,
                 downsample=3,
                 return_motion=False,
                 return_digit=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample = downsample
        self.return_label = return_label

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
            split = 'MovingMNIST/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test':
            split = 'MovingMNIST/test_split.csv'
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')
        self.video_info = video_info

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.seq_len*self.downsample <= 0:
            return [None]
        n = 1
        start_idx = np.random.choice(
            range(vlen-self.seq_len*self.downsample), n)
        seq_idx = np.arange(self.seq_len)*self.downsample + start_idx
        return [seq_idx, vpath]

    def __getitem__(self, index):
        vpath, vlen, motion_id, digit = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.seq_len)

        seq = [pil_loader(os.path.join(vpath, '{i}.jpg'))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)

        if self.return_motion and self.return_digit:
            motion = torch.LongTensor([motion_id])
            digit = torch.LongTensor([digit])
            return t_seq, motion, digit
        if self.return_motion and not self.return_digit:
            motion = torch.LongTensor([motion_id])
            return t_seq, motion
        if not self.return_motion and self.return_digit:
            digit = torch.LongTensor([digit])
            return t_seq, digit
        return t_seq, digit

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
