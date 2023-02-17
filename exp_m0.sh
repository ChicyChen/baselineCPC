#!/bin/bash

checkpoints='ckpt_normalized'
bbepoch=100
# ssl need ~1000 epochs in SimCRL
ftepoch=60
# start_epoch=30
# ft takes ~60 epochs in SimCRL

batchsize=16
downsample=3

split=1
ft_split=1

pred_step=5
nsub=5

datasetssl='ucf'
datasetft='hmdb'


for loss_mode in 0 1 2 3 4
do 
    # ssl
    # 1layer_2dGRU_static_M0
    python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --dataset $datasetssl
    python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --seeall --dataset $datasetssl

    # ft
    # 1layer_2dGRU_static_M0
    bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
    python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft
    bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
    python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft

    # # ft with pretrain
    # # 1layer_2dGRU_static_M0
    # bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
    # prefolder="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch""_split$ft_split"
    # pretrain="$prefolder""/epoch$start_epoch"".pth.tar"
    # python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --pretrain $pretrain --start-epoch $start_epoch
    # bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize"
    # prefolder="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch""_split$ft_split"
    # pretrain="$prefolder""/epoch$start_epoch"".pth.tar"
    # python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --pretrain $pretrain --start-epoch $start_epoch
    

    # test
    # 1layer_2dGRU_static_M0
    bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch""_split$ft_split"
    python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft
    bbfoler="$checkpoints""/ucf240_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.0001_wd1e-05_bs$batchsize""/finetune_hmdb240_lr0.001_wd0.0001_ep$bbepoch""_split$ft_split"
    python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft
    
done

