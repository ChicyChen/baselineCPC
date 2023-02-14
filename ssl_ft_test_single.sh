#!/bin/bash

bbepoch=100
# ssl need ~1000 epochs in SimCRL
ftepoch=30
# ft takes ~60 epochs in SimCRL
batchsize=16
downsample=3
pred_step=5
nsub=5
split=1
test_split=1


# ssl

# 1layer_2dGRU_static
python ssl_learning.py --epochs $bbepoch --model 1 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# 1layer_2dGRU_static, sa
python ssl_learning.py --epochs $bbepoch --model 1 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize

# ft

# 1layer_2dGRU_static
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 1
# 1layer_2dGRU_static, sa
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 1

# test

# 1layer_2dGRU_static
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $split --model 1
# 1layer_2dGRU_static, sa
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $split --model 1

# ssl

# # 2layer_2dGRU_static_R1, sa
# python ssl_learning.py --epochs $bbepoch --model 6 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize

# ft

# # 2layer_2dGRU_static_R1, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 6

# test

# # 2layer_2dGRU_static_R1, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 6

pred_step=5
nsub=5
lada=0.5

# ssl

# # 2layer_2dGRU_static_R1, sa
# python ssl_learning.py --epochs $bbepoch --model 6 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --lada $lada
# # 2layer_2dGRU_static_R1
# python ssl_learning.py --epochs $bbepoch --model 6 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --lada $lada

# ft

# # 2layer_2dGRU_static_R1, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada$lada""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 6 --lada $lada
# # 2layer_2dGRU_static_R1
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada$lada""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 6 --lada $lada

# ssl

# 2layer_2dGRU_static_R1, sa
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada$lada""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 6 --lada $lada
# 2layer_2dGRU_static_R1
bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada$lada""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 6 --lada $lada


