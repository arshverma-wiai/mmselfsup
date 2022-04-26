#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_50k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_pt_50k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_in_nih_50k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_68k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_pt_68k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_in_nih_68k.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_v2a.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_nih_pt_v2a.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 taskset --cpu-list 0-79 zsh tools/dist_train.sh configs/selfsup/simclr/simclr_resnet50_8xb128-coslr-500e_in_nih_v2a.py 8