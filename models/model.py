# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: model.py
@time: 2021/11/2 14:08
"""
import torch
import torch.nn as nn
import numpy as np
from models.hrt import HighResolutionTransformer
from models.swin_transformer import SwinTransformer
import yaml

path = "../config/hrt_base.yaml"
config = yaml.load(open(path,'r'),yaml.SafeLoader)['MODEL']['HRT']

class HR_SwinNet(nn.Module):
    def __init__(self):
        super(HR_SwinNet, self).__init__()
        self.HR_rgb_branch = HighResolutionTransformer(config, 1000)
        self.Swin_depth_branch = SwinTransformer()

    def forward(self,x,d):
        x_result = self.HR_rgb_branch(x)
        d_result = self.Swin_depth_branch(d)
        print(len(x_result))
        print(len(d_result))

if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    d = torch.randn(1,3,224,224)
    model = HR_SwinNet()
    model(x, d)