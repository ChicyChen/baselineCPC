#!/bin/bash

# ssl
# SSLEPOCH=10
# # 2layer_2dGRU_static_B1, sa
# python ssl_learning.py --epochs $SSLEPOCH --model 2 --seeall
# # 2layer_2dGRU_static_B2, sa
# python ssl_learning.py --epochs $SSLEPOCH --model 3 --seeall
# # 2layer_2dGRU_static_A1, sa
# python ssl_learning.py --epochs $SSLEPOCH --model 4 --seeall
# # 2layer_2dGRU_static_A2, sa
# python ssl_learning.py --epochs $SSLEPOCH --model 5 --seeall
# # 2layer_2dGRU_static_B1
# python ssl_learning.py --epochs $SSLEPOCH --model 2
# # 2layer_2dGRU_static_B2
# python ssl_learning.py --epochs $SSLEPOCH --model 3
# # 2layer_2dGRU_static_A1
# python ssl_learning.py --epochs $SSLEPOCH --model 4
# # 2layer_2dGRU_static_A2
# python ssl_learning.py --epochs $SSLEPOCH --model 5

# ft or freeze
bbepoch=10
ftepoch=10

# 2layer_2dGRU_static_B1, sa
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_saTrue_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 2
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 2
# 2layer_2dGRU_static_B2, sa
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B2_uoTrue_saTrue_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 3
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 3
# 2layer_2dGRU_static_A1, sa
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_A1_uoTrue_saTrue_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 4
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 4
# 2layer_2dGRU_static_A2, sa
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_A2_uoTrue_saTrue_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 5
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 5

# 2layer_2dGRU_static_B1
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_saFalse_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 2
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 2
# 2layer_2dGRU_static_B2
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_B2_uoTrue_saFalse_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 3
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 3
# 2layer_2dGRU_static_A1
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_A1_uoTrue_saFalse_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 4
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 4
# 2layer_2dGRU_static_A2
bbfoler="checkpoints/ucf240_split0_2layer_2dGRU_static_A2_uoTrue_saFalse_lr0.0001_wd1e-05_bs32"
python action_recognation.py --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 5
python action_recognation.py --freeze --backbone_folder $bbfoler --backbone_epoch $bbepoch --epochs $ftepoch --model 5