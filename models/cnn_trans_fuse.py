# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: cnn_trans_fuse.py
@time: 2021/11/26 10:50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class CTFuse(nn.Module):
    def __init__(self, RGB_channel, Depth_channel):
        super(CTFuse, self).__init__()
        self.channel_attention = ChannelAttention(RGB_channel, Depth_channel) # 出条状
        self.spatial_attention = SpatialAttention()
        self.SAconv_1 = conv3x3(Depth_channel, RGB_channel)
        self.sigmoid = nn.Sigmoid()
        self.conv_bn_relu1 = conv3x3_bn_relu(RGB_channel+Depth_channel, RGB_channel)
        self.depth_conv2 = conv3x3(Depth_channel, RGB_channel)
        self.rgb_conv2 = conv3x3(RGB_channel,RGB_channel)
        self.outCBR = conv3x3_bn_relu(RGB_channel, RGB_channel)
    def forward(self, x, d):

        d_shape = d.shape # B,C,H,W
        x_shape = x.shape # B,C,H,W

        d_conv1 = self.SAconv_1(d)
        d_conv_up = F.interpolate(input=d_conv1, size=x_shape[2:], mode="bilinear", align_corners=True)
        d_conv_up_sigmoid = self.sigmoid(d_conv_up)
        rgb_sa = torch.mul(x, d_conv_up_sigmoid)

        r_camap = self.channel_attention(x)
        depth_ca = torch.mul(d, r_camap)
        depth_ca_up = F.interpolate(input = depth_ca, size=x_shape[2:], mode="bilinear", align_corners=True)

        rgb_depth_cat = torch.cat((rgb_sa, depth_ca_up), dim=1)

        rd_c_b_r = self.conv_bn_relu1(rgb_depth_cat)

        depth_conv2 = self.depth_conv2(d)
        depth_conv_up = F.interpolate(depth_conv2, x_shape[2:], mode="bilinear", align_corners=True)

        R_conv = self.rgb_conv2(x)

        three_sum_out = depth_conv_up + rd_c_b_r + R_conv

        out = self.outCBR(three_sum_out)

        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)

        res += x
        return res

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes,ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, out_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

