# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 20:57
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import glob
import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
cmap = plt.cm.jet


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='NYUDepth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-b', '--batch-size', default=2, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--max_iter', default=9000000, type=int, metavar='miter')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--data_dir", type=str, default="/data/liuyx/nyuv2",
                        help="the root folder of dataset")
    parser.add_argument("--data_list", type=str, default="/data/liuyx/nyuv2/nyu_depth_v2_train_fill_every10/train_rgb.txt",
                        help="the root folder of dataset")
    parser.add_argument("--test_dir", type=str, default="/data/liuyx/nyuv2/nyu_depth_v2_labeled",
                        help="the root folder of dataset")
    parser.add_argument("--test_list", type=str, default="./dataset/list/nyuv2/test.txt",
                        help="the root folder of dataset")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--input-size', default=(353,257), type=list,
                         help='print frequency (default: 10)')
    parser.add_argument('--random-mirror',action='store_true',
                         help='whether to randomly mirror the inputs')
    parser.add_argument('--save-dir',type=str,default='baseline',
                         help='whether to randomly mirror the inputs')
    args = parser.parse_args()
    return args


def get_output_directory(args):
    save_dir = os.path.join ('./run', args.save_dir)
    return save_dir


# 保存检查点
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def adjust_learning_rate(optimizer, lr_init, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('*********************************************')
    print('  epoch[%d], learning rate = %f' % (epoch, lr))
    print('*********************************************')


# ploy策略的学习率更新
def update_ploy_lr(optimizer, initialized_lr, current_step, max_step, power=0.9):
    lr = initialized_lr * ((1 - float(current_step) / max_step) ** (power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred,img_mean):
    rgb =  np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0)) +img_mean # H, W, C
    rgb =cv2.resize(rgb,(640,480))
    depth_target_cpu=np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = (depth_pred.data.cpu().numpy())
    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
