import cv2
import torch
from PIL.Image import Image
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from torch.nn import functional as F
import json


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def ImageResize(img: np.ndarray, h, w) -> np.ndarray:
    """
        将图像等比缩放，并输出ndarry格式
        :param img: 输入图像（numpy.ndarry）
        :param h: 需要得到的图像的高
        :param w: 需要得到的图像的宽
        :return: 变化后得到的图像
        """
    # 获取图像大小参数
    if len(img.shape) == 3:
        height, width, bytesPerComponent = img.shape
    else:
        height, width = img.shape
    # 计算各边缩放比例
    raw_rate, need_rate = height / float(width), h / float(w)
    # 计算需要添加边框的大小
    h_adds, w_adds = int((width * (h / float(w)) - height) / 2.0) if raw_rate <= need_rate else 0, int((height / (h / float(w)) - width) / 2.0) if raw_rate > need_rate else 0
    # 给图片添加灰边
    value = [0, 0, 0] if len(img.shape) == 3 else [0]
    img = cv2.copyMakeBorder(img, h_adds, h_adds, w_adds, w_adds, cv2.BORDER_CONSTANT, value=value)
    # 图像转换
    img = cv2.resize(img, (w, h))
    return img


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


def ImageTransformToQPixmap(img: np.ndarray, w, h) -> QPixmap:
    """
    将图像等比缩放，并输出qpixmap格式
    :param img: 输入图像（numpy.ndarry）
    :param w: 需要得到的图像的宽
    :param h: 需要得到的图像的高
    :return: 变化后得到的图像
    """
    # 获取图像大小参数
    height, width, bytesPerComponent = img.shape
    # 计算各边缩放比例
    raw_rate, need_rate = height / float(width), h / float(w)
    # 计算需要添加边框的大小
    h_adds, w_adds = int((width * (h / float(w)) - height) / 2.0) if raw_rate <= need_rate else 0, int((height / (h / float(w)) - width) / 2.0) if raw_rate > need_rate else 0
    # 给图片添加灰边
    img = cv2.copyMakeBorder(img, h_adds, h_adds, w_adds, w_adds, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # 图像转换
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    img = cv2.resize(img, (w, h))
    height, width, bytesPerComponent = img.shape
    bytesPerLine = bytesPerComponent * width
    image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(image)


def GetSource(pred, label):
    """
    计算图像得分
    即计算每个像素点的正误
    得分 = 正确的像素点数量 / 所有像素点
    :param pred: 预测结果
    :param label: 标签
    :return:
    """
    b, class_num, h, w = pred.size()

    pred_temp = F.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous(), -1)
    pred_map = pred_temp.cpu().detach().numpy().argmax(axis=-1)
    correct = torch.eq(torch.from_numpy(pred_map), label.cpu()).float().sum().item()

    return correct / (b * h * w)


def LoadConfigFile(path='./config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    return config
