import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from binary_fractions import TwosComplement
# import torch.optim as optim
from torchvision import datasets, transforms, models
# from torch.utils.data.dataloader import Dataloader
# from torch.autograd import Variable
# from torchviz import make_dot
from matplotlib import pyplot as plt
# from matplotlib.gridspec import GridSpec
import numpy as np
import random
from datetime import datetime
import time
import pdb
from self_models import *
import sys
import os
import shutil
import argparse

def shift_and_return(number, bit):
    return int(number * (2**(bit-1))) / (2**(bit-1))

def flip_bits(number, start_bit, end_bit):
    random.seed(time.time())
    # print(number)
    number = int(float(number) * (2**15))
    number = TwosComplement(number, length=16)
    binary_tmp = str(number)
    # print(binary_tmp)
    position = random.randint(start_bit, end_bit)
    # print(position)
    if binary_tmp[position] == '0':
        binary_tmp = binary_tmp[:position] + '1' + binary_tmp[position+1:]
    elif binary_tmp[position] == '1':
        binary_tmp = binary_tmp[:position] + '0' + binary_tmp[position+1:]
    # print(binary_tmp)
    number = int(TwosComplement.to_float(binary_tmp))
    # print(number)
    return number / (2**15)

def stuck_bits(number, start_bit, end_bit, stuck):
    number = int(float(number) * (2**15))
    number = TwosComplement(number, length=16)
    binary_tmp = str(number)
    # print(binary_tmp)
    for i in range(start_bit, end_bit+1):
        if stuck == 0:
            binary_tmp = binary_tmp[:i] + '0' + binary_tmp[i+1:]
        elif stuck == 1:
            binary_tmp = binary_tmp[:i] + '1' + binary_tmp[i+1:]
    # print(binary_tmp)
    number = int(TwosComplement.to_float(binary_tmp))
    # print(number)
    return number / (2**15)


def remove_bits(number, start_bit, end_bit):
    # print(number)
    number = int(float(number) * (2**15))
    if number >= 0:
        number = TwosComplement(number, length=16)
        binary_tmp = str(number)
        # print(binary_tmp)
        for i in range(start_bit, end_bit+1):
            binary_tmp = binary_tmp[:i] + '0' + binary_tmp[i+1:]
            # print(binary_tmp)
        number = int(TwosComplement.to_float(binary_tmp))
    else:
        number = TwosComplement(number, length=16)
        binary_tmp = str(number)
        # print(binary_tmp)
        for i in range(start_bit, end_bit+1):
            binary_tmp = binary_tmp[:i] + '1' + binary_tmp[i+1:]
            # print(binary_tmp)
        number = int(TwosComplement.to_float(binary_tmp))
            
    # print(number)
    return number / (2**15)

def insert_faults(state, key, frate, error_type, start_bit, end_bit, stuck_bit=1):
    weight_size = list(state['state_dict'][key].size())
    n_index = len(weight_size)
    n_elem  = torch.numel(state['state_dict'][key])
    n_faults = int(n_elem*frate)
    dimension = len(state['state_dict'][key].size())
    i = 0
    if error_type == 'remove_bits':
        for d1 in range(weight_size[0]):
            for d2 in range(weight_size[1]):
                if dimension > 2:
                    for d3 in range(weight_size[2]):
                        if dimension > 3:
                            for d4 in range(weight_size[3]):
                                state['state_dict'][key][d1,d2,d3,d4] = remove_bits(state['state_dict'][key][d1,d2,d3,d4], start_bit, end_bit)
                        else:
                            state['state_dict'][key][d1,d2,d3] = remove_bits(state['state_dict'][key][d1,d2,d3], start_bit, end_bit)
                else:
                    state['state_dict'][key][d1,d2] = remove_bits(state['state_dict'][key][d1,d2], start_bit, end_bit)
    else:
        while (i < n_faults):
            rand_index = np.random.randint(0, high=n_elem, size=n_index)
            # print(rand_index)
            findexes = tuple(np.remainder(rand_index, weight_size))
            # print(findexes)
            i += 1
            if error_type == 'stuck_bits':
                state['state_dict'][key].data[findexes] = stuck_bits(state['state_dict'][key].data[findexes], start_bit, end_bit, stuck_bit)
            elif error_type == 'flip_bits':
                state['state_dict'][key].data[findexes] = flip_bits(state['state_dict'][key].data[findexes], start_bit, end_bit)

    return state

def float_to_fix_point(state, key, fixed_bits):
    weight_size = list(state['state_dict'][key].size())
    dimension = len(state['state_dict'][key].size())
    print(dimension)
    if dimension > 0:
        print(weight_size[0])
    if dimension > 1:
        print(weight_size[1])
    if dimension > 2:
        print(weight_size[2])
    if dimension > 3:
        print(weight_size[3])

    print(state['state_dict'][key][0,0])
    for d1 in range(weight_size[0]):
        for d2 in range(weight_size[1]):
            if dimension > 2:
                for d3 in range(weight_size[2]):
                    if dimension > 3:
                        for d4 in range(weight_size[3]):
                            state['state_dict'][key][d1,d2,d3,d4] = shift_and_return(state['state_dict'][key][d1,d2,d3,d4], fixed_bits)
                    else:
                        state['state_dict'][key][d1,d2,d3] = shift_and_return(state['state_dict'][key][d1,d2,d3], fixed_bits)
            else:
                state['state_dict'][key][d1,d2] = shift_and_return(state['state_dict'][key][d1,d2], fixed_bits)
                        

    print(state['state_dict'][key][0,0])
    return state

def main():
    fp_type = 'fp16'
    fixed_bits = 16
    # error_type = 'remove_bits'
    error_type = 'stuck_bits'
    # error_type = 'flip_bits'
    # error_type = ''
    frate = 0.001
    start_bit = 12
    end_bit = 15
    model = VGG_SNN_STDB(vgg_name = 'VGG16',
                         activation = 'Linear',
                         labels = 10,
                         timesteps = 200,
                         leak = 1.0,
                         default_threshold = 1.0,
                         alpha = 0.3,
                         beta = 0.01,
                         dropout = 0.3,
                         kernel_size = 3,
                         dataset='CIFAR10')
    model = nn.DataParallel(model)
    pretrained_snn = './trained_models/snn/snn_vgg16_cifar10.pth'
    state = torch.load(pretrained_snn, map_location='cpu')

    cur_dict = model.state_dict()
    for key in state['state_dict'].keys():
        if key in cur_dict:
            if (state['state_dict'][key].shape == cur_dict[key].shape):
                state = float_to_fix_point(state, key, fixed_bits)
                state = insert_faults(state, key, frate, error_type, start_bit, end_bit)
                print('\n Loaded {} from {}'.format(key, pretrained_snn))
            else:
                print('\n Size mismatch {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model.state_dict()[key].shape))
        else:
            print('\n Loaded weight {} not present in current model'.format(key))

    filename = './trained_models/snn/snn_vgg16_cifar10_'+fp_type+error_type+'.pth'
        
    torch.save(state, filename)

if __name__ == '__main__':
    main()
    
