# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Rao Fu, RainbowSecret
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .hrt import HighResolutionTransformer


def build_model(config):
    model_type = config.MODEL.TYPE

    model = HighResolutionTransformer(
        config.MODEL.HRT, num_classes=config.MODEL.NUM_CLASSES)

    print(model)
    return model
