import torch
import torch.nn as nn


class MobileNetV2Block(nn.Module):
    """
    MobileNetV2 Block
    一共分为三个阶段：利用1x1卷积核对通道数进行修正 -> 3x3卷积特征提取 -> 1x1卷积核修正
    输入参数：
        in_channel: 输入通道
        out_channel: 输出通道
        expand_rate: 膨胀系数
        kernel_size: 卷积核大小
        stride: 步长
        padding: padding
    """
    def __init__(self, in_channel: int, out_channel: int, expand_rate: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, inplace=False):
        super(MobileNetV2Block, self).__init__()
        # 获取膨胀后的通道数
        expand_channel = expand_rate * in_channel
        # 是否采用残差连接
        self.res_connect = True if stride == 1 else False
        # 卷积模块
        self.conv = nn.Sequential(
            # 阶段一：利用1x1卷积核对通道进行修正
            nn.Conv2d(in_channel, expand_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(expand_channel),
            nn.ReLU(inplace=inplace),
            # 阶段二：利用3x3卷积核进行特征提取
            nn.Conv2d(expand_channel, expand_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(expand_channel),
            nn.ReLU(inplace=inplace),
            # 阶段三：利用1x1卷积核进行通道数修正
            nn.Conv2d(expand_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )
        # 残差连接模块
        self.res_connect_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
        )

    def forward(self, x):
        # 主干卷积层
        out = self.conv(x)
        # 残差连接层
        if self.res_connect:
            out = out + self.res_connect_conv(x)
        return out


class MobileNet(nn.Module):
    """
    MobileNet特征提取网络
    输入：[3, 512, 512]
    输出：
        浅层特征：[24, 128, 128]
        深层特征：[320, 32, 32]
    """

    def __init__(self, input_channel=3, inplace=False):
        super(MobileNet, self).__init__()
        # 头部预处理网络
        self.pre_head = nn.Sequential(
            # 3, 512, 512 -> 3, 512, 512
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=inplace),
            # 3, 512, 512 -> 3, 512, 512
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=inplace),
            # 3, 512, 512 -> 3, 256, 256
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=inplace),
        )
        # 获取主干特征提取网络
        in_c = 16          # 输入通道
        self.features = []       # 主干特征提取网络
        self.feature_setting = [
            # 设置主干特征提取网络参数，这里采用16倍下采样
            # num, out_channel, expand_rate, stride
            [1, 24, 6, 1],          # 16, 256, 256 -> 24, 256, 256
            [2, 24, 6, 2],          # 24, 256, 256 -> 24, 128, 128  -> 浅层特征层
            [3, 32, 6, 2],          # 24, 128, 128 -> 32, 64, 64
            [4, 64, 6, 2],          # 32, 64, 64   -> 64, 32, 32
            [3, 96, 6, 1],          # 64, 32, 32   -> 96, 32, 32
            [3, 160, 6, 2],         # 96, 32, 32   -> 160, 16, 16
            [1, 320, 6, 1],         # 160, 16, 16  -> 320, 16, 16   -> 深层特征层
        ]
        for n, c, e, s in self.feature_setting:
            out_c = c       # 获取输出通道
            for i in range(n):
                # 如果有多组卷积层，则仅第一个采用stride，后面都采用默认的stride
                if i == 0:
                    self.features.append(MobileNetV2Block(in_c, out_c, e, stride=s, inplace=inplace))
                else:
                    self.features.append(MobileNetV2Block(in_c, out_c, e, stride=1, inplace=inplace))
                in_c = out_c
        self.shallow_features = self.features[:3]   # 浅层特征提取
        self.shallow_features = nn.Sequential(*self.shallow_features)
        self.deep_features = self.features[3:]  # 深层特征提取
        self.deep_features = nn.Sequential(*self.deep_features)

    def forward(self, x):
        # 图像头部预处理
        pre = self.pre_head(x)
        # 主干特征提取网络
        shallow_feature = self.shallow_features(pre)
        deep_feature = self.deep_features(shallow_feature)
        return shallow_feature, deep_feature


if __name__ == '__main__':
    net = MobileNet()
    x = torch.randn([2, 3, 256, 256])
    shallow, deep = net(x)
    print(net)
    print(f"shallow size: {shallow.size()}   deep size: {deep.size()}")
