#!/bin/bash

# define variables
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B2_uoTrue_lr0.0001_wd1e-05_bs32"
# bbfolder_nossl = "checkpoints/ucf240_split0_1layer_2dGRU_static_nossl"
bbepoch=100
ftepoch=10

# finetune after ssl, hmdb
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch
# freeze after ssl, hmdb
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch
# finetune after ssl, ucf
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --dataset ucf240
# freeze after ssl, ucf
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --dataset ucf240

# finetune no ssl, ucf
# python action_recognation.py --backbone_folder $bbfolder_nossl --epochs $ftepoch
# # freeze no ssl, ucf
# python action_recognation.py --freeze --backbone_folder $bbfolder_nossl --epochs $ftepoch
# # finetune no ssl, ucf
# python action_recognation.py --backbone_folder $bbfolder_nossl --epochs $ftepoch --dataset ucf240
# # freeze no ssl, ucf
# python action_recognation.py --freeze --backbone_folder $bbfolder_nossl --epochs $ftepoch --dataset ucf240