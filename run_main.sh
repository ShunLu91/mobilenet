#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6  &&  python  -u  main.py --exp_name mobilenetv2 \
--epochs 100 --weight_decay 0.0001 --learning_rate 0.1 --batch_size 256 \
--val_interval 5 > log_mobilenetv2 2>&1 &