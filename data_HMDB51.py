import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2

from tqdm import tqdm
from joblib import Parallel, delayed

from utils import *
from augmentation import *


def get_data_hmdb(transform=None, mode='train', num_seq=20, downsample=3, which_split=1, return_label=False, batch_size=16, dim=150):
    print('Loading data for "%s" ...' % mode)
    dataset = HMDB51(mode=mode,
                        transform=transform,
                        num_seq=num_seq,
                        downsample=downsample,
                        which_split=which_split,
                        return_label=return_label,
                        dim=dim
                        )
    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


class HMDB51(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 num_seq=20,
                 downsample=3,
                 which_split=1,
                 return_label=True,
                 dim=150
                 ):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.dim = dim

        if dim == 150:
            folder_name = 'hmdb51'
        else:
            folder_name = 'hmdb51_240'

        # splits
        if mode == 'train':
            if self.which_split == 0:
                split = 'data/'+folder_name+'/train.csv'
            else:
                split = 'data/'+folder_name+'/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            if self.which_split == 0:
                split = 'data/'+folder_name+'/test.csv'
            else:
                split = 'data/'+folder_name+'/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            _, vlen, _ = row
            if vlen-self.num_seq*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        print("Droped number of videos:", len(drop_idx))

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        '''sample index from a video'''
        if vlen-self.num_seq*self.downsample <= 0: raise ValueError('video too short')
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.downsample), n)
        seq_idx = np.arange(self.num_seq)*self.downsample + start_idx
        return [seq_idx, vpath]


    def __getitem__(self, index):
        vpath, vlen, aid = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform

        (C, H, W) = t_seq[0].size()

        # print(C, H, W)

        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, C, H, W)

        if self.return_label:
            label = torch.LongTensor([aid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)


if __name__ == '__main__':
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        Scale(size=(224,224)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])
    train_loader = get_data_hmdb(transform, 'train', dim=240)

