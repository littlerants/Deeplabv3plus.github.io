import ResNet_atrous as resnet
import torch
import torch.nn as nn
import torch.nn.functional as func


class Block(nn.Module):
    def __init__(self, inplane, plane, kernel_size=1, stride=1, atrous=1, Relu=None):
        super(Block, self).__init__()
        if kernel_size == 1:
            self.conv1 = nn.Conv2d(inplane, plane, kernel_size=kernel_size, padding=0, stride=stride)
        else:
            self.conv1 = nn.Conv2d(inplane, plane, kernel_size=kernel_size, padding=atrous, stride=stride, dilation=atrous)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.Relu = Relu

    def forward(self, x):

        out = self.conv1(x)
        if out.size(-1) > 1 and out.size(-2) > 1:
            out = self.bn1(out)
        if self.Relu is None:
            out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, inplane, plane, rate_list):
        super(ASPP, self).__init__()
        self.branch1 = Block(inplane, plane, kernel_size=1, stride=1, atrous=rate_list[0])
        self.branch2 = Block(inplane, plane, kernel_size=3, stride=1, atrous=rate_list[1])
        self.branch3 = Block(inplane, plane, kernel_size=3, stride=1, atrous=rate_list[2])
        self.branch4 = Block(inplane, plane, kernel_size=3, stride=1, atrous=rate_list[3])
        # 全局平均池化层
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = Block(inplane, plane, kernel_size=1, stride=1, atrous=1)
        self.convcat = Block(plane * 5, plane, kernel_size=1, stride=1, atrous=1)
        self.branch5 = nn.Sequential(self.branch5_avg, self.branch5_conv,)

    def forward(self, x):
        conv1x1 = self.branch1(x)
        conv3x3r6 = self.branch2(x)
        conv3x3r12 = self.branch3(x)
        conv3x3r18 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = func.interpolate(global_feature, (x.size(-2), x.size(-1)), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3r6, conv3x3r12, conv3x3r18, global_feature], dim=1)
        result = self.convcat(feature_cat)
        return result


class DeepLabv3Plus(nn.Module):
    def __init__(self, ):
        super(DeepLabv3Plus, self).__init__()
        self.backbone = resnet.ResNetAtrous('resnet50', resnet.Bottleneck, [3,5,6,3], 16)
        inplane = 2048
        self.aspp = ASPP(inplane, 256, [1, 6, 12, 18])
        self.dropout1 = nn.Dropout2d(0.5)
        self.upsampleby4 = nn.UpsamplingBilinear2d(scale_factor=4)
        indim =256
        self.skip_conv = nn.Sequential(
            nn.Conv2d(indim, 64, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.class_conv = nn.Conv2d(256, 8, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        layers = self.backbone(x)
        out = self.aspp(layers[-1])
        out = self.dropout1(out)
        out = self.upsampleby4(out)
        feature_shallow = self.skip_conv(layers[0])
        featurecat = torch.cat([out, feature_shallow], dim=1)
        result = self.conv_cat(featurecat)
        result = self.class_conv(result)
        result = self.upsampleby4(result)
        return result

# the function 'main()' is test program, you can ignore...


def main():
    f = False
    if f:   # ASPP
        x = torch.rand((1, 3, 1, 1))
        conv = ASPP(3, 6, [1, 6, 12, 18])
        y = conv(x)
        print(y.size())
    else:   # Deeplabv3plus
        deeplab = DeepLabv3Plus()
        z = torch.rand((1, 3, 256, 512))
        w = deeplab(z)
        print(w.size())


if __name__ == '__main__':
    main()



