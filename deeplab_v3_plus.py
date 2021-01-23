"""
@ Author: ryanreadbooks
@ Time: 2021/1/23
@ File name: deeplab_v3_plus.py
@ File description: define the whole network here
"""

import torch
import torch.nn.functional as F

from resnet import *
from decoder import Decoder
from aspp import ASPP


class DeepLabV3Plus(nn.Module):
    """
    The deeplab model
    """

    def __init__(self, num_classes, backbone='50'):
        super(DeepLabV3Plus, self).__init__()
        if backbone == '50':
            self.backbone = resnet50()
        elif backbone == '101':
            self.backbone = resnet101()
        elif backbone == '152':
            self.backbone = resnet152()
        else:
            raise ValueError('backbone {} not supported.'.format(backbone))

        self.num_classes = num_classes
        self.aspp = ASPP()
        self.decoder = Decoder(self.num_classes)

    def forward(self, input_x):
        """
        The forward process
        :param input_x: input of the whole network
        :return:
        """

        x_feat, low_level_feature = self.backbone(input_x)
        x_feat = self.aspp(x_feat)
        x_feat = self.decoder(x_feat, low_level_feature)
        out = F.interpolate(x_feat, size=input_x.size()[2:], mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    deeplab_model = DeepLabV3Plus(num_classes=1, backbone='50').cuda()
    in_x = torch.randn(2, 3, 480, 640).cuda()
    print(deeplab_model(in_x).shape)
