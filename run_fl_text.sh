#!/bin/bash

python main.py \
  --model bert \
  --epochs 5 \
  --local_ep 5 \
  --dataset sst2 \
  --tuning lora \
  --num_classes 2 \
  --num_users 20 \
  --frac 0.4 \
  --attackers 0.3 \
  --attack_type addSent \
  --lr 1e-4 \
  --optimizer adamw \
  --gpu \
  --defense krum \
  --poison_ratio 1.0