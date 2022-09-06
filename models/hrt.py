# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: model.py
@time: 2021/11/22 13:50
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.bottleneck_block import Bottleneck, BottleneckDWP
from models.modules.transformer_block import GeneralTransformerBlock
from models.cnn.cnn_resnet import resnet18
from models.decoder_fuse.transformer_dec_fuse import TransFuseModel
import yaml
from thop import profile, clever_format
from models.MCGFuse import MCG
import numpy as np
from models.cnn_trans_fuse import CTFuse

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


blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
}


BN_MOMENTUM = 0.1


class HighResolutionTransformerModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        num_input_resolutions,
        attn_types,
        ffn_types,
        multi_scale_output=True,
        drop_paths=0.0,

    ):
        """
        Args:
            num_heads: the number of head witin each MHSA
            num_window_sizes: the window size for the local self-attention
            num_input_resolutions: the spatial height/width of the input feature maps.
        """
        super(HighResolutionTransformerModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
        )

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.attn_types = attn_types
        self.ffn_types = ffn_types

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_input_resolutions,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        attn_types,
        ffn_types,
        drop_paths,
        stride=1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                input_resolution=num_input_resolutions[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                attn_type=attn_types[branch_index][0],
                ffn_type=ffn_types[branch_index][0],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    input_resolution=num_input_resolutions[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    attn_type=attn_types[branch_index][i],
                    ffn_type=ffn_types[branch_index][i],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_input_resolutions,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        attn_types,
        ffn_types,
        drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):

        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels


    def forward(self, x):
        d = x[-1]
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        # TODO len(self.fuse_layers) = self.num_branches
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    # TODO fuse select
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionTransformer(nn.Module):
    def __init__(self, cfg, num_classes=1000, **kwargs):
        super(HighResolutionTransformer, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stochastic depth
        depth_s2 = cfg["STAGE2"]["NUM_BLOCKS"][0] * cfg["STAGE2"]["NUM_MODULES"]
        depth_s3 = cfg["STAGE3"]["NUM_BLOCKS"][0] * cfg["STAGE3"]["NUM_MODULES"]
        depth_s4 = cfg["STAGE4"]["NUM_BLOCKS"][0] * cfg["STAGE4"]["NUM_MODULES"]
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = cfg["DROP_PATH_RATE"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0] # 64
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0] # 2

        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]] # GeneralTransformerBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2],

        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2 : depth_s2 + depth_s3],

        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            drop_paths=dpr[depth_s2 + depth_s3 :],

        )
        self.depth_back = resnet18()
        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            pre_stage_channels
        )

        self.classifier = nn.Linear(2048, num_classes)

        self.fuse1 = MCG(256, 64, 56, 56)
        self.fuse2 = MCG(156, 128, 28, 28)
        self.fuse3 = MCG(312, 256, 14, 14)
        self.fuse4 = MCG(624, 512, 7, 7)

        # deocder
        self.stand_cahnnel1 = conv1x1_bn_relu(78, 128)
        self.stand_cahnnel2 = conv1x1_bn_relu(156, 128)
        self.stand_cahnnel3 = conv1x1_bn_relu(312, 128)
        self.stand_cahnnel4 = conv1x1_bn_relu(624, 128)

        self.TransTo4e = TransFuseModel(num_blocks=2, x_channels=128, nx=49, y_channels=128, ny=4116)
        self.TransTo3e = TransFuseModel(num_blocks=2, x_channels=128, nx=196, y_channels=128, ny=3969)
        self.TransTo2e = TransFuseModel(num_blocks=2, x_channels=128, nx=784, y_channels=128, ny=3381)
        self.TransTo1e = TransFuseModel(num_blocks=2, x_channels=128, nx=3136, y_channels=128, ny=1029)

        self.pred_linear = nn.Linear(128, 1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def _make_head(self, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = [32, 64, 128, 256]

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer) # 3，[78,156,312]
        num_branches_pre = len(num_channels_pre_layer) # 2, [78,156]

        transition_layers = []
        for i in range(num_branches_cur): # 0,1,2
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]

                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        input_resolution=None,
        num_heads=1,
        stride=1,
        window_size=7,
        halo_size=1,
        mlp_ratio=4.0,
        q_dilation=1,
        kv_dilation=1,
        sr_ratio=1,
        attn_type="msw",
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    halo_size,
                    mlp_ratio,
                    q_dilation,
                    kv_dilation,
                    sr_ratio,
                    attn_type,
                )
            )
        else:
            layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    # TODO _make_stage
    def _make_stage(
        self, layer_config, num_inchannels, multi_scale_output=True, drop_paths=0.0,depth_in_channels=256,
    ):

        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        num_heads = layer_config["NUM_HEADS"]
        num_window_sizes = layer_config["NUM_WINDOW_SIZES"]
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]
        num_input_resolutions = layer_config["NUM_RESOLUTIONS"]
        attn_types = layer_config["ATTN_TYPES"]
        ffn_types = layer_config["FFN_TYPES"]

        modules = []
        for i in range(num_modules):

            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionTransformerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    num_input_resolutions,
                    attn_types[i],
                    ffn_types[i],
                    reset_multi_scale_output,
                    drop_paths=drop_paths[num_blocks[0] * i : num_blocks[0] * (i + 1)],

                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, d):
        d = self.depth_back(d) # [B, C, H, W]
        d1,d2,d3,d4, = d # 128,256,512,1024,1024


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = self.fuse1(x, d1)

        x_list = []

        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        x_list[-1] = self.fuse2(x_list[-1], d2)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x_list[-1] = self.fuse3(x_list[-1], d3)
        y_list = self.stage3(x_list)


        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x_list[-1] = self.fuse4(x_list[-1], d4)
        y_list = self.stage4(x_list)

        x1,x2,x3,x4 = y_list

        x1 = self.stand_cahnnel1(x1).flatten(2).permute(0, 2, 1)
        x2 = self.stand_cahnnel2(x2).flatten(2).permute(0, 2, 1)
        x3 = self.stand_cahnnel3(x3).flatten(2).permute(0, 2, 1)
        x4 = self.stand_cahnnel4(x4).flatten(2).permute(0, 2, 1)

        x4e = self.TransTo4e(x4, torch.cat((x1, x2, x3), dim=1))
        x3e = self.TransTo3e(x3, torch.cat((x1, x2, x4e), dim=1))
        x2e = self.TransTo2e(x2, torch.cat((x1, x3e, x4e), dim=1))
        x1e = self.TransTo1e(x1, torch.cat((x2e, x3e, x4e), dim=1))

        pred = self.pred_linear(x1e)
        B, N, C = pred.shape
        out = pred.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        out = self.up4(out)

        return out

    def init_weights(
        self,
        hr_pretrained="",
        depth_pretrained=""
    ):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(hr_pretrained):
            pretrained_dict = torch.load(hr_pretrained)['model']

            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                print("=> loading {} pretrained model {}".format(k, hr_pretrained))

            model_dict.update(pretrained_dict)
            if depth_pretrained is not None:

                load_dict = torch.load(depth_pretrained)
                new_state_dict = {k: v for k, v in load_dict.items() if k not in[ "conv1.weight"]}
                for k, _ in load_dict.items():
                    print("=>ResNet18 loading {} pretrained model {}".format(k, hr_pretrained))
                self.depth_back.load_state_dict(new_state_dict, strict=False)

            self.load_state_dict(model_dict)

if __name__ == '__main__':
    path = "../config/hrt_base.yaml"
    a = torch.randn(1, 3, 224, 224)
    b = torch.randn(1, 1, 224, 224)
    config = yaml.load(open(path, "r"),yaml.SafeLoader)['MODEL']['HRT']
    hr_pth_path = r""
    cnn_pth_path = r""
    model = HighResolutionTransformer(config, 1000)
    model.init_weights(hr_pth_path, cnn_pth_path)

    # out = model(a, b)
    flops, params = profile(model, inputs=(a,b))
    flops, params = clever_format([flops, params], "%.2f")

    print(params, flops)

