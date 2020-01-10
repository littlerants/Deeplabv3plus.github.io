import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.utils.model_zoo as model_zoo
'''**************ResNet*******************'''


class BlockAtrous(nn.Module):
    def __init__(self, in_plane, plane, kernel_size=3, atrous=1, stride=1, Relu=None, normal=None):
        super(BlockAtrous, self).__init__()
        if kernel_size == 1:
            # 1x1卷积单独定义
            self.conv1 = nn.Conv2d(in_plane, plane, kernel_size=1, padding=0, stride=stride)
        elif normal is not None:
            self.conv1 = nn.Conv2d(in_plane, plane, kernel_size=kernel_size, padding=int(kernel_size/2), stride=stride)
        else:
            self.conv1 = nn.Conv2d(in_plane, plane, kernel_size=kernel_size, padding=atrous, stride=stride, dilation=atrous)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.Relu = Relu

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.Relu is None:
            out = self.relu(out)
        return out


class BasicBlockAtrous(nn.Module):
    expansion = 1

    def __init__(self, in_plane, plane, atrous=1, stride=1, downsample=None):
        super(BasicBlockAtrous,self).__init__()
        self.conv1 = BlockAtrous(in_plane, plane, 3, atrous, stride)
        self.conv2 = BlockAtrous(plane, plane, True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, plane, atrous=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = BlockAtrous(in_plane, plane, 1, 0, 1)
        self.conv2 = BlockAtrous(plane, plane, 3, atrous, stride)
        self.conv3 = BlockAtrous(plane, plane * Bottleneck.expansion, 1, 0, 1, True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class ResNetAtrous(nn.Module):
    def __init__(self, name, block, list, output_stride=16):
        super(ResNetAtrous, self).__init__()
        stride_listv= []
        if output_stride == 16:
            stride_list = [2, 2, 1]
        if output_stride == 8:      # 原文作者不建议使用8以及以下的outputstride 原因：1. 网络结构所致 2. GPU内存限制
            stride_list = [2, 1, 1]
        self.name = name
        self.in_plane = 64
        # model
        self.conv1 = BlockAtrous(3, self.in_plane, kernel_size=7, stride=2, normal=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers1 = self.make_layers(block, 64, list[0])
        self.layers2 = self.make_layers(block, 128, list[1], stride=stride_list[0])
        self.layers3 = self.make_layers(block, 256, list[2], stride=stride_list[1], atrous=(16//output_stride))
        # output_stride = 16
        self.layers4 = self.make_layers(block, 512, list[3], stride=stride_list[2], atrous=2)
        self.layers5 = self.make_layers(block, 512, list[3], stride=stride_list[2], atrous=4)
        self.layers6 = self.make_layers(block, 512, list[3], stride=stride_list[2], atrous=8)

    def make_layers(self, block, plane, nums, stride=1, atrous=1):
        downsample = None
        if stride != 1 or self.in_plane != plane*block.expansion:
            downsample = nn.Sequential(
                BlockAtrous(self.in_plane, plane * block.expansion, kernel_size=1, stride=stride, Relu=True)
            )
        layers = []

        layers.append(block(self.in_plane, plane, atrous=atrous, stride=stride, downsample=downsample))
        self.in_plane = plane * block.expansion
        for i in range(1, nums):
            layers.append(block(self.in_plane, plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        layers = []
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layers1(out)
        layers.append(out)
        out = self.layers2(out)
        layers.append(out)
        out = self.layers3(out)
        layers.append(out)
        out = self.layers4(out)
        out = self.layers5(out)
        out = self.layers6(out)
        layers.append(out)
        return layers
#  *************the main() is the test program, you can ignore**********


def main():
    x = torch.rand((1, 3, 224, 224))
    conv2 = ResNetAtrous('resnet101', Bottleneck, [3, 5, 6, 3], 16)
    y = conv2(x)

    for i in range(len(y)):
        print(y[i].size())


# the program is beginning here

if __name__ =='__main__':
    main()


