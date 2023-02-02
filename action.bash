#!/bin/bash

# train ssl for 100 epchs
python ssl_learning.py --useout --epochs 100 --pretrain checkpoints/ucf240_split0_1layer_2dGRU_static_uoTrue_lr0.0001_wd1e-05_bs32/epoch10.pth.tar --start-epoch 10

# finetune after ssl, hmdb
python action_recognation.py --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_uoTrue_lr0.0001_wd1e-05_bs32 --backbone_epoch 100
# freeze after ssl, hmdb
python action_recognation.py --freeze --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_uoTrue_lr0.0001_wd1e-05_bs32 --backbone_epoch 100

# finetune after ssl, ucf
python action_recognation.py --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_uoTrue_lr0.0001_wd1e-05_bs32 --backbone_epoch 100 --dataset ucf240
# freeze after ssl, ucf
python action_recognation.py --freeze --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_uoTrue_lr0.0001_wd1e-05_bs32 --backbone_epoch 100 --dataset ucf240

# finetune after ssl, ucf
python action_recognation.py --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_lr0.0001_wd1e-05_bs32 --backbone_epoch 10 --dataset ucf240
# freeze after ssl, ucf
python action_recognation.py --freeze --backbone_folder checkpoints/ucf240_split0_1layer_2dGRU_static_lr0.0001_wd1e-05_bs32 --backbone_epoch 10 --dataset ucf240