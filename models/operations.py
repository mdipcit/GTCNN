
import torch
import torch.nn as nn
from torch.nn import Module

class Conv(nn.Module):
    def __init__(self, C_in, C_out,use_bnorm=True):
        super(Conv, self).__init__()
        if use_bnorm:
            self.op = nn.Sequential(nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(C_out, eps=0.0001, momentum = 0.95),
                        nn.ReLU(inplace=False)
                        )
        else:
            self.op = nn.Sequential(nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=False)
                        )
    def forward(self, x):
        return self.op(x)
