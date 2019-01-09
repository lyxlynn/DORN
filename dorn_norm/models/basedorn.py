#import encoding.nn as nn
#import encoding.functions as F
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import models.depth2normal_torch as d2n
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

from modules import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
#current_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, os.path.join(current_dir, '../../inplace_abn'))



#BatchNorm2d = nn.BatchNorm2d

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    
    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class Decoder_Module(nn.Module):

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=False)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(48),
            nn.ReLU(inplace=False)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 1024, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(1024),
            nn.ReLU(inplace=False)
            )
        # self.RC1 = Residual_Covolution(512, 1024, num_classes)
        # self.RC2 = Residual_Covolution(512, 1024, num_classes)
        # self.RC3 = Residual_Covolution(512, 1024, num_classes)
        self.conv4 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()

        xt = F.upsample(self.conv1(xt), size=(h, w), mode='bilinear')
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x  
   
 
class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        if torch.cuda.is_available():
            decode_label = torch.zeros((N, 1, H, W), dtype=torch.float32).cuda()
            ord_labels = torch.zeros((N, C // 2, H, W), dtype=torch.float32).cuda()
        else:
            decode_label = torch.zeros((N, 1, H, W), dtype=torch.float32)
            ord_labels = torch.zeros((N, C // 2, H, W), dtype=torch.float32)
        # print('#1 decode size:', decode_label.size())
        ord_num = C // 2
        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()
        # print('A size:', A.size())
        # print('B size:', B.size())

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)

        ord_c = nn.functional.softmax(C, dim=1)

        # print('C size:', C.size())
        # print('ord_c size:', ord_c.size())

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        decode_c = torch.sum(ord_c1>=0.5, dim=1).view(-1, 1, H, W).float()
        # print('ord_c1 size:', ord_c1.size())

       # print('decode_label size:', decode_label.size())
       # print('decode_label size:', decode_label)
        return decode_c, ord_c1
class ResNet_base(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64, affine = affine_par)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))
      #  self.layer5 = PSPddModule(2048, 512)
      #  self.edgelayer = Edge_Module(256, 2) 
        self.delayer = Decoder_Module(num_classes)
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=False),
            )
        self.layer6=nn.Conv2d(512, 160, kernel_size=1, padding=0, stride=1, bias=True)
        self.orl =OrdinalRegressionLayer()
            

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        print(multi_grid)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        _,_,h,w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
       # x5 = self.delayer(x5)   #added
        x5 = self.layer5(x5)
       # out1,x = self.layer6(x5, x2)
        out2 = self.layer6(x5)     
        depth = F.upsample(out2, size=(h, w), mode='bilinear')
        depth_labels,ord_labels = self.orl(depth)
        return depth_labels,ord_labels
         
def basedorn(num_classes=160):

    model = ResNet_base(Bottleneck,[3, 4, 23, 3], num_classes)

    return model 
