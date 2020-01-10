import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.utils.model_zoo as model_zoo
'''**************ResNet*******************'''
version = 0


class Block(nn.Module):
    def __init__(self, in_plane, plane, kernel_size=3, padding=1, stride=1, atrous=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, plane, kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_plane, plane, stride=1,atrous=1, down_sample=None ):
        super(BasicBlock, self).__init__()
        self.block = Block(in_plane, plane, kernel_size=3, stride=stride, atrous=atrous)

        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, padding=1, stride=atrous)
        self.bn2 = nn.BatchNorm2d(plane)
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        x += identity
        if version:
            x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, plane, stride=1, atrous=1, down_sample=None):
        super(Bottleneck, self).__init__()

        self.block1 = Block(in_plane, plane, 1, padding=0, stride=1)
        self.block2 = Block(plane, plane, 3, stride=stride, atrous=atrous)
        self.conv1 = nn.Conv2d(plane, plane*self.expansion, 1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(plane*self.expansion)
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        x += identity
        if version:
            x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers_list, num_class):
        super(ResNet, self).__init__()
        self.inplane = 64
        self.dilation = 1

        self.block = block

        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers1 = self.mak_layer(64, layers_list[0], block)
        self.layers2 = self.mak_layer(128, layers_list[1], block, stride=2)
        self.layers3 = self.mak_layer(256, layers_list[2], block, stride=2)
        self.layers4 = self.mak_layer(512, layers_list[3], block, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_class)

    def mak_layer(self, plane, num, block, stride=1):
        downsample = None
        if stride != 1 or self.inplane != plane*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, plane*block.expansion, kernel_size=1, padding=0, stride=stride),
                nn.BatchNorm2d(plane*block.expansion))
        layers = []
        layers.append(block(self.inplane, plane, stride=stride, atrous=1, down_sample=downsample))
        self.inplane = plane*block.expansion
        for i in range(1, num):
            layers.append(block(self.inplane, plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
#  *************the main() is the test program, you can ignore**********


def main():
    x = torch.rand((1, 3, 224, 224))
    # conv1= Bottleneck(32, 8)
    conv2 = ResNet(Bottleneck, [3, 4, 6, 3],  100)
    # y = conv1(x)
    z = conv2(x)
    print('output:\t', z.size())

# the program is beginning here


if __name__ =='__main__':
    main()


