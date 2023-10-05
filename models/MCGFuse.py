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

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class MCG(nn.Module):
    def __init__(self, rgb_inchannels, depth_inchannels,h,w):
        super(MCG, self).__init__()
        self.channels = rgb_inchannels
        self.convDtoR = nn.Conv2d(depth_inchannels, rgb_inchannels, 3,1,1)
        self.convTo2 = nn.Conv2d(rgb_inchannels*2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()

        self.h = h
        self.w = w
        self.coordAttention = CoordAtt(rgb_inchannels, rgb_inchannels, self.h, self.w)

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

        Gm_out = self.coordAttention(Gm)
        out = Gm_out + Ga

        return out

if __name__ == '__main__':
    a = torch.randn(1,78,56,56)
    b = torch.randn(1,128,56,56)
    model = MCG(78,128)
    out = model(a,b)
    print(out.shape)