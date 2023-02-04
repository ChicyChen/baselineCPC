#!/bin/bash

bbepoch=10
ftepoch=10

# 2layer_2dGRU_static_B1
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_saFalse_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --model 2
# 2layer_2dGRU_static_B1, sa
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_saTrue_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --model 2