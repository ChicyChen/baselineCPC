#!/bin/bash

SSLEPOCH=10
# 2layer_2dGRU_static_B1
python ssl_learning.py --epochs $SSLEPOCH --model 2 --seeall
# 2layer_2dGRU_static_B2
python ssl_learning.py --epochs $SSLEPOCH --model 3 --seeall

# # ssl, model 1
# python ssl_learning.py --epochs $SSLEPOCH --model 1
# # ssl, model 2
# python ssl_learning.py --epochs $SSLEPOCH --model 2 --pretrain checkpoints/ucf240_split0_2layer_2dGRU_static_B1_uoTrue_lr0.0001_wd1e-05_bs32/epoch10.pth.tar --start-epoch 10
# # ssl, model 3
# python ssl_learning.py --epochs $SSLEPOCH --model 3 --pretrain checkpoints/ucf240_split0_2layer_2dGRU_static_B2_uoTrue_lr0.0001_wd1e-05_bs32/epoch10.pth.tar --start-epoch 10
# # ssl, model 4
# python ssl_learning.py --epochs $SSLEPOCH --model 4 --useout
