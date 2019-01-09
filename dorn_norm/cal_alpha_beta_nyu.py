# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/25 11:15
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import os

import torch

import numpy as np
from torch.utils import data
#from dataloaders import nyu_dataloader
from dataset.datasets import LIPParsingEdgeDataSet, LIPDataValSet
import utils
alpha = np.inf
beta = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
args = utils.parse_command()

def NYUDepth_loader(batch_size=32, isTrain=True):
    global args
    if isTrain:
        train_loader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.max_iter, crop_size=args.input_size, 
                     mirror=args.random_mirror), 
                    batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
        return train_loader
    else:
        val_loader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=args.input_size), 
                                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        return val_loader

batch_size = 64

train_loader = NYUDepth_loader( batch_size, isTrain=True)
val_loader = NYUDepth_loader( batch_size, isTrain=False)

for i, (input, target,_,_) in enumerate(train_loader):
    input, target = input.cuda(), target.cuda()
    print('train ', i)
    valid_mask = (target > 0).detach()
    max = torch.max(target[valid_mask])
    min = torch.min(target[valid_mask])

    if alpha > min:
        alpha = min
        print(min)

    if beta < max:
        beta = max

for i, (input, target,_,_) in enumerate(val_loader):
    input, target = input.cuda(), target.cuda()
    print('val ', i)
    valid_mask = (target > 0).detach()
    max = torch.max(target[valid_mask])
    min = torch.min(target[valid_mask])

    if alpha > min:
        alpha = min

    if beta < max:
        beta = max

print(alpha, beta)



