#!/bin/bash

# finetune from scratch
python action_recognation.py --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_nossl

# finetune after ssl
python action_recognation.py
python action_recognation.py --epochs 20 --start-epoch 10 --pretrain checkpoints/ucf240_split0_1layer_2dGRU_static_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep10/epoch10.pth.tar
python action_recognation.py --epochs 30 --start-epoch 20 --pretrain checkpoints/ucf240_split0_1layer_2dGRU_static_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep10/epoch20.pth.tar

# freeze after ssl
python action_recognation.py --freeze