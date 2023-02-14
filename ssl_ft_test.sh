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

# # 1layer_2dGRU_static, sa
# python ssl_learning.py --epochs $bbepoch --model 1 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_B1, sa
# python ssl_learning.py --epochs $bbepoch --model 2 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_B2, sa
# python ssl_learning.py --epochs $bbepoch --model 3 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_A1, sa
# python ssl_learning.py --epochs $bbepoch --model 4 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_A2, sa
# python ssl_learning.py --epochs $bbepoch --model 5 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_R1, sa
# python ssl_learning.py --epochs $bbepoch --model 6 --seeall --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize

# # 1layer_2dGRU_static
# python ssl_learning.py --epochs $bbepoch --model 1 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_B1
# python ssl_learning.py --epochs $bbepoch --model 2 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_B2
# python ssl_learning.py --epochs $bbepoch --model 3 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_A1
# python ssl_learning.py --epochs $bbepoch --model 4 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_A2
# python ssl_learning.py --epochs $bbepoch --model 5 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize
# # 2layer_2dGRU_static_R1
# python ssl_learning.py --epochs $bbepoch --model 6 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize

# ft

# # 1layer_2dGRU_static, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 1
# # 2layer_2dGRU_static_B1, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 2
# # 2layer_2dGRU_static_B2, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B2_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 3
# # 2layer_2dGRU_static_A1, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 4
# # 2layer_2dGRU_static_A2, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A2_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 5
# # 2layer_2dGRU_static_R1, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 6

# # 1layer_2dGRU_static
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 1
# # 2layer_2dGRU_static_B1
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 2
# # 2layer_2dGRU_static_B2
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B2_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 3
# # 2layer_2dGRU_static_A1
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 4
# # 2layer_2dGRU_static_A2
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A2_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 5
# # 2layer_2dGRU_static_R1
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
# python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $split --model 6

# test
# # 1layer_2dGRU_static, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 1
# # 2layer_2dGRU_static_B1, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 2
# # 2layer_2dGRU_static_B2, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B2_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 3
# # 2layer_2dGRU_static_A1, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 4
# # 2layer_2dGRU_static_A2, sa
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A2_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 5
# # 2layer_2dGRU_static_R1, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 6

# # 1layer_2dGRU_static, sa
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 1
# # 2layer_2dGRU_static_B1
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 2
# # 2layer_2dGRU_static_B2
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_B2_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 3
# # 2layer_2dGRU_static_A1
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 4
# # 2layer_2dGRU_static_A2
# bbfoler="checkpoints/ucf240_split$split""_2layer_2dGRU_static_A2_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 5
# # 2layer_2dGRU_static_R1
# bbfoler="checkpoints/ucf240_split$split""_1layer_2dGRU_static_R1_lada0.1_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch"
# python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $test_split --model 6