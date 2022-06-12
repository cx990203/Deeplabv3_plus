import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import *


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    空洞空间池化卷积
    """
    def __init__(self, in_channel, out_channel, inplace=False):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=6, dilation=6, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=12, dilation=12, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=18, dilation=18, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel * 5, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )

    def forward(self, x):
        b, c, row, col = x.size()
        # 分支1-4：采用空洞卷积
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        branch4 = self.conv4(x)
        # 分支5： 全局平均池化+卷积
        branch5 = torch.mean(x, 2, True)
        branch5 = torch.mean(branch5, 3, True)
        branch5 = self.conv5(branch5)
        branch5 = F.interpolate(branch5, (row, col), None, 'bilinear', True)
        # 将五个特征层进行合成
        feature_cat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        result = self.conv(feature_cat)
        return result


class Deeplabv3(nn.Module):
    """
    Deeplabv3+网络
    """
    def __init__(self, class_num, input_size, inplace=True):
        super(Deeplabv3, self).__init__()
        self.input_size = input_size
        # 采用MobileNet作为主干特征提取网络
        self.backcone = MobileNet(inplace=inplace)
        # ASPP
        self.aspp = ASPP(320, 256)
        # 浅层特征卷积处理
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=inplace)
        )
        # 输出卷积层
        self.out_layer = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace),
            nn.Dropout(0.1),
            nn.Conv2d(256, class_num, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # 特征提取
        shallow, deep = self.backcone(x)
        # ASPP空洞卷积，获取加强特征层
        deep = self.aspp(deep)
        # 浅层卷积层进行通道扩展
        shallow = self.shallow_conv(shallow)
        # 将深层特整层进行上采样，然后和浅层特整层进行合成
        b, c, row, col = shallow.size()
        deep = F.interpolate(deep, (row, col), None, 'bilinear', True)
        feature = torch.cat([shallow, deep], dim=1)
        # 对特征进行最后变换，然后进行上采样，得到结果
        out = self.out_layer(feature)
        out = F.interpolate(out, (self.input_size[0], self.input_size[1]),  None, 'bilinear', True)
        return out


if __name__ == '__main__':
    input_size = [256, 256]
    net = Deeplabv3(2, input_size)
    x = torch.randn([2, 3, *input_size])
    pre = net(x)
    print(pre.size())
