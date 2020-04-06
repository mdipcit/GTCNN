#!/usr/bin/env python
# -*- coding: utf-8 -*-


#    Copyright 2020 Kaito Imai

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .operations import *
import numpy as np

def double_conv(in_channels, out_channels,num_cbr=2,use_bnorm=True):
    #DCBR
    layers = []
    if use_bnorm:
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1,bias=True))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(num_cbr-1):
        if use_bnorm:
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=True))
        layers.append(nn.ReLU(inplace=True))
    return  nn.Sequential(*layers)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upmodule=True,attention=False,num_cbr=2,use_bnorm=True,concat_type='concat'):
        super().__init__()
        self.upmodule = upmodule
        self.concat_type = concat_type
        if self.concat_type ==  'sum':
            in_channels = in_channels //2

        if upmodule == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_channels, out_channels,num_cbr,use_bnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.concat_type ==  'sum':
            x = x2+x1
        else:
            x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels=64,num_cbr=2,pooling ='maxpool',use_bnorm=True):
        super().__init__()
        if pooling == 'maxpool':
            self.subpixel= nn.MaxPool2d(2)
            self.maxpool_conv = nn.Sequential(
                double_conv(in_channels, out_channels,num_cbr,use_bnorm)
            )


    def forward(self, x):
        return self.maxpool_conv(self.subpixel(x))
        
class GTL(nn.Module):

    def __init__(self,netconf={}, GTL_IC=1,GTL_OC=64,n_chan=64,num_cbr=2,act='ReLU',pooling='maxpool',upmodule='bilinear',stage=4,use_bnorm=True,stage_option='slim',concat_type='concat',**kwargs):
        super(GTL,self).__init__()
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

        elif act == 'identity':
            self.act = nn.Sequential()
        self.stage_option = stage_option
        self.stage=stage
        self.lambdas= np.array([0,0,0,0,0]).astype(np.float32) # lambdas for modulation
        if stage_option == 'slim':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
            elif self.stage == 2:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 0:
                self.out=double_conv(n_chan,GTL_OC ,num_cbr,use_bnorm)

            elif self.stage == 3:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:

                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 5:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 6:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up6 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down6 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)

        elif stage_option == 'outconv_slim':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            self.out = nn.Sequential(nn.Conv2d(in_channels=n_chan, out_channels=GTL_OC, kernel_size=1, padding=0))
            
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 3:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 5:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 6:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up6 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down6 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)

            elif self.stage == 0:
                self.dconv= nn.Sequential(
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm)
            )
        elif stage_option == 'fat':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:
                self.up1 = Up(n_chan*16+n_chan*8, n_chan*8, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*8+n_chan*4, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4+n_chan*2, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2+n_chan, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan, n_chan*2,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan*2, n_chan*4,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan*4, n_chan*8,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan*8, n_chan*16,num_cbr=num_cbr,use_bnorm=use_bnorm)

        elif stage_option == 'shuffle':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up4 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
            elif self.stage == 3:
                self.up3 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up1 = Up(n_chan*8, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                # self.up1 = Up(n_chan*16, n_chan*8,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down3 = Down(n_chan*16, n_chan*8, num_cbr=num_cbr*2,use_bnorm=use_bnorm,pooling =pooling)
                # self.down4 = Down(n_chan*32, n_chan*16, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)

            elif self.stage == 4:
                self.up4 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*8, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up1 = Up(n_chan*16, n_chan*8,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down3 = Down(n_chan*16, n_chan*8, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down4 = Down(n_chan*32, n_chan*16, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
        
    def forward(self, x):
        x1 = self.input(x)

            
        if self.stage == 1:
            x = self.down1(x1)
            x = self.up4(x, x1)
        elif self.stage == 0:
            x = self.out(x)
        elif self.stage == 2:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
        elif self.stage == 3:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        elif self.stage == 4:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            ########################################################
            # Modification
            if any(self.lambdas != 0):
                print('##################')
                print('Texture modulation')
                print(self.lambdas)
                print('###################')
                x1,x2,x3,x4,x5 =x1+self.lambdas[0] ,x2+self.lambdas[1],\
                    x3+self.lambdas[2],x4+self.lambdas[3],x5+self.lambdas[4]
            ##############################################
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        elif self.stage == 5:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        elif self.stage == 6:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x = self.up1(x7, x6)
            x = self.up2(x, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)


        if self.stage_option =='outconv_slim':
            x = self.out(x)
        out = x
        return self.act(out)