"""
@ Author: ryanreadbooks
@ Time: 2021/1/23
@ File name: resnet.py
@ File description: define the resnet backbone here, resnet-50, resnet-101 and resnet-152 supported
"""

import torch
import torch.nn.modules as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Class for building resnet
    """

    def __init__(self, block, layers, BatchNorm=None):
        super(ResNet, self).__init__()
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        self._BatchNorm = BatchNorm

        # resnet head
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # middle
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        blocks = [1, 2, 4]
        self.layer4 = self._make_multi_grid_layer(block, 512, blocks=blocks, stride=1, dilation=2)
        # self.layer5 = self._make_multi_grid_layer(block, 512, blocks=blocks, stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        Make resnet layer
        :param block: Bottleneck
        :param planes:
        :param blocks:
        :param stride:
        :param dilation:
        :return:
        """
        BatchNorm = self._BatchNorm
        downsample = None
        # need conv in shortcut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.in_planes, planes, stride, downsample, dilation))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_multi_grid_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        Multi-grid unit
        :param block: Bottleneck
        :param planes:
        :param blocks:
        :param stride:
        :param dilation:
        :return:
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.in_planes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.in_planes, planes, stride=1,
                                dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        return x, low_level_features

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    model = resnet50()
    print(model)
    input = torch.rand(1, 3, 480, 640)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
