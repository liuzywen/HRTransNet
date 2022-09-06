# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/11/25 5:12
"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--model_name', type=str, default="HRTransNet", help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=17, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
parser.add_argument('--hr_load', type=str, default="./hrt_base.pth", help='train from checkpoints')

parser.add_argument('--cnn_load', type=str, default="./resnet18-5c106cde.pth", help='train from checkpoints')

parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='', help='can same as depth path, but not used')
parser.add_argument('--test_rgb_root', type=str, default='', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='', help='the test gt images root')
parser.add_argument('--test_edge_root', type=str, default='', help='not used')

parser.add_argument('--save_path', type=str, default='./HRTransNet/', help='the path to save models and logs')
opt = parser.parse_args()