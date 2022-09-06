# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: transformer_dec_fuse.py
@time: 2021/12/8 01:00
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import time
from models.decoder_fuse.transformer_block import get_sinusoid_encoding
from thop import profile, clever_format

class EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(self.in_channels, self.key_channels)
        self.queries = nn.Linear(self.in_channels, self.key_channels)
        self.values = nn.Linear(self.in_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.in_channels)

    def forward(self, input_, x_pos_embed):
        B,N,C = input_.size()
        assert C == self.in_channels,"C {} != inchannels {}".format(C, self.in_channels)
        assert input_.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(input_.shape, x_pos_embed.shape)
        keys = self.keys(input_ + x_pos_embed).permute(0, 2, 1) #.reshape((n, self.key_channels, h * w))
        queries = self.queries(input_ + x_pos_embed).permute(0, 2, 1) #.reshape(n, self.key_channels, h * w)
        values = self.values(input_).permute(0, 2, 1)#.reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1,2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value

class Multi_EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, x_channels, y_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.queries = nn.Linear(self.x_channels, self.key_channels)
        self.keys = nn.Linear(self.y_channels, self.key_channels)
        self.values = nn.Linear(self.y_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.x_channels)


    def forward(self, x, y, x_pos_embed, y_pos_embed):
        Bx,Nx,Cx = x.size()
        assert Cx == self.x_channels,"Cx {} != inchannels {}".format(Cx, self.x_channels)
        assert x.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(x.shape, x_pos_embed.shape)
        By, Ny, Cy = y.size()
        assert Cy == self.y_channels, "Cy {} != inchannels {}".format(Cy, self.y_channels)
        assert y.shape[1:] == y_pos_embed.shape[1:], "y.shape {} != y_pos_embed.shape {}".format(y.shape, y_pos_embed.shape)

        queries = self.queries(x + x_pos_embed).permute(0, 2, 1) #.reshape(n, self.key_channels, h * w)
        keys = self.keys(y + y_pos_embed).permute(0, 2, 1)  # .reshape((n, self.key_channels, h * w))
        values = self.values(y).permute(0, 2, 1)#.reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]

            context = key @ value.transpose(1, 2)

            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1, 2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,x_channels, nx, y_channels, ny):
        super(Block, self).__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.nx = nx
        self.ny = ny
        self.x_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels), requires_grad=False)
        self.y_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels), requires_grad=False)
        self.x2_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels),requires_grad=False)
        self.norm_layer = nn.LayerNorm(x_channels)
        # nn.LayerNorm

        self.self_attn = EfficientAttention(x_channels, x_channels, 4, x_channels)
        self.cross_attn = Multi_EfficientAttention(x_channels=x_channels, y_channels=y_channels, key_channels=x_channels, head_count=4, value_channels=x_channels)
        self.mlp = Mlp(in_features=x_channels, hidden_features=x_channels * 4,out_features= x_channels)
    def forward(self,x, y):
        x_atten = self.self_attn(x, self.x_pos_embed)
        Osa = self.norm_layer(x + x_atten)
        xy_attn = self.cross_attn(Osa, y, self.x2_pos_embed, self.y_pos_embed)
        Oca = self.norm_layer(xy_attn + Osa)
        Of = self.mlp(Oca)
        Oo = self.norm_layer(Of + Oca)
        return Oo

class TransFuseModel(nn.Module):
    def __init__(self, num_blocks, x_channels, nx, y_channels, ny):
        super(TransFuseModel, self).__init__()
        assert x_channels == y_channels, "channel_X-{} should same as channel_Y-{}".format(x_channels, y_channels)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            # todo
            Block(x_channels=x_channels, nx=nx, y_channels=y_channels, ny=ny)for i in range(self.num_blocks)
        ])
        self.norm =nn.LayerNorm(x_channels)

    def forward(self,x,y):
        '''
        :param x: shape B,Nx,C
        :param y: shape B,Ny,C
        :return: shape B,Nx,c
        '''
        # Bx, Nx, Cx = x.shape
        # By, Ny, Cy = y.shape
        for block in self.blocks:
            x = block(x, y)
        x = self.norm(x)
        return x
