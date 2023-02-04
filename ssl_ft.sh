#!/bin/bash

bbepoch=1
# ssl need ~1000 epochs in SimCRL
ftepoch=1
# ft takes ~60 epochs in SimCRL
downsample=3
pred_step=5
nsub=5

# # ssl

# # 2layer_2dGRU_static_R1, sa
# python ssl_learning.py --epochs $bbepoch --model 6 --seeall
# # 2layer_2dGRU_static_R1
# python ssl_learning.py --epochs $bbepoch --model 6

# ft

# 2layer_2dGRU_static_R1, sa
bbfoler="checkpoints/ucf240_split0_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 6
# 2layer_2dGRU_static_R1
bbfoler="checkpoints/ucf240_split0_1layer_2dGRU_static_R1_lada0.1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 6

# test

# 2layer_2dGRU_static_R1, sa
bbfoler="checkpoints/ucf240_split0_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --model 6
# 2layer_2dGRU_static_R1
bbfoler="checkpoints/ucf240_split0_1layer_2dGRU_static_R1_lada0.1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs32/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --model 6