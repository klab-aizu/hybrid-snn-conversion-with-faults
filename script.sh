#!/bin/bash

python snn.py --architecture VGG11 --learning_rate 1e-4 --epochs 50 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --dataset CIFAR100 --batch_size 64 --optimizer Adam --timesteps 50 --leak 1.0 --scaling_factor 0.7 --dropout 0.3 --kernel_size 3 --pretrained_ann './trained_models/ann/ann_vgg11_cifar100.pth' --log 