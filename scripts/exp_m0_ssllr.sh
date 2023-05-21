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

gpu='0,1'

loss_mode=0

ftlr=0.001


# # ssl
# # 1layer_2dGRU_static_M0
# python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --dataset $datasetssl --gpu $gpu --prefix $checkpoints --lr 1e-5

# ft
# 1layer_2dGRU_static_M0
bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.001_wd1e-05_bs$batchsize"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu --lr $ftlr


# test
# 1layer_2dGRU_static_M0
bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.001_wd1e-05_bs$batchsize""/finetune_$datasetft""_lr0.001""_wd0.0001_ep$bbepoch""_split$ft_split"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu



# # ssl
# # 1layer_2dGRU_static_M0
# python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --dataset $datasetssl --gpu $gpu --prefix $checkpoints --lr 1e-6

# ft
# 1layer_2dGRU_static_M0
bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.01_wd1e-05_bs$batchsize"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu --lr $ftlr


# test
# 1layer_2dGRU_static_M0
bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr0.01_wd1e-05_bs$batchsize""/finetune_$datasetft""_lr0.001""_wd0.0001_ep$bbepoch""_split$ft_split"
python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu


for ssllr in 0.0005 0.0003
do 
    # ssl
    # 1layer_2dGRU_static_M0
    python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --dataset $datasetssl --gpu $gpu --prefix $checkpoints --lr $ssllr
    # python ssl_learning.py --epochs $bbepoch --model 7 --downsample $downsample --pred_step $pred_step --nsub $nsub --which_split $split --batch_size $batchsize --loss_mode $loss_mode --seeall --dataset $datasetssl --gpu $gpu --prefix $checkpoints --lr $ssllr

    # ft
    # 1layer_2dGRU_static_M0
    bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr$ssllr""_wd1e-05_bs$batchsize"
    python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu --lr $ftlr
    # bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr$ssllr""_wd1e-05_bs$batchsize"
    # python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu --lr $ftlr
    

    # test
    # 1layer_2dGRU_static_M0
    bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saFalse_ds$downsample""_ps$pred_step""_ns$nsub""_lr$ssllr""_wd1e-05_bs$batchsize""/finetune_$datasetft""_lr0.001""_wd0.0001_ep$bbepoch""_split$ft_split"
    python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu
    # bbfoler="$checkpoints""/$datasetssl""_split$split""_1layer_2dGRU_static_M0_loss$loss_mode""_uoTrue_saTrue_ds$downsample""_ps$pred_step""_ns$nsub""_lr$ssllr""_wd1e-05_bs$batchsize""/finetune_$datasetft""_lr0.001""_wd0.0001_ep$bbepoch""_split$ft_split"
    # python action_test.py --backbone_folder $bbfoler --backbone_epoch $ftepoch --which_split $ft_split --model 7 --dataset $datasetft --gpu $gpu
    
done