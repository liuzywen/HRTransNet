# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: MGFuse.py
@time: 2021/12/6 9:31
"""
import torch
import torch.nn as nn
from models.coordatt import CoordAtt
import numpy as np

class MCG(nn.Module):
    def __init__(self, rgb_inchannels, depth_inchannels,h,w):
        super(MCG, self).__init__()
        self.channels = rgb_inchannels
        self.convDtoR = nn.Conv2d(depth_inchannels, rgb_inchannels, 3,1,1)
        self.convTo2 = nn.Conv2d(rgb_inchannels*2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(self.channels, self.channels // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(self.channels // 16, self.channels, 1, bias=False)

        self.MlpConv = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.h = h
        self.w = w

    def forward(self,r,d):
        d = self.convDtoR(d)
        d = self.relu(d)
        H = torch.cat((r,d), dim=1)
        H_conv = self.convTo2(H)
        H_conv = self.sig(H_conv)
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        Ga = r * ga
        Gm = d * gm

        GmA = self.global_avg_pool(Gm)

        GmA_fc = self.fc2(self.relu(self.fc1(GmA)))
        GmA_fc = self.sig(GmA_fc)
        Gm1 = Gm * GmA_fc

        Gm1M = self.global_max_pool(Gm1)
        Gm1M_conv = self.MlpConv(Gm1M)
        Gm2 = self.sig(Gm1M_conv)

        Gm_out = Gm1 * Gm2
        # Gm_out = self.coordAttention(Gm)
        out = Gm_out + Ga

        return out

if __name__ == '__main__':
    a = torch.randn(1,78,56,56)
    b = torch.randn(1,128,56,56)
    model = MCG(78,128)
    out = model(a,b)
    print(out.shape)