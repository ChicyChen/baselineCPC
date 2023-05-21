#!/bin/bash

bbepoch=10
# 2layer_2dGRU_static_B1
python ssl_learning.py --epochs $bbepoch --model 2 --seeall
# 2layer_2dGRU_static_B2
python ssl_learning.py --epochs $bbepoch --model 3 --seeall
# 2layer_2dGRU_static_A1
python ssl_learning.py --epochs $bbepoch --model 4 --seeall
# 2layer_2dGRU_static_A2
python ssl_learning.py --epochs $bbepoch --model 5 --seeall


