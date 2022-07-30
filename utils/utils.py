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


def IouSource(pred, label) -> list:
    """
    计算图像得分，类似于IOU计算
    得分 = 类别像素点数量交集 / 类别像素点并集
    :param pred: 预测结果
    :param label: 标签
    :return:
    """
    b, class_num, h, w = pred.size()
    pred_temp = F.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous(), -1)
    pred_map = pred_temp.cpu().detach().numpy().argmax(axis=-1)
    source = []
    for i in range(class_num - 1):
        class_idx = np.ones((h, w)) * i + 1             # 判断的类别索引标号
        label_use = label.cpu().numpy().copy()          # tensor -> numpy
        if len(pred_map.shape) == 3 and len(label_use.shape) == 2:
            # 有时候如果是单张预测的话，会出现维度不一致的问题
            pred_map = pred_map[0, :, :]
        intersection = np.sum((label_use == class_idx) & (pred_map == class_idx))       # 计算交集长度
        union = np.sum((label_use == class_idx) | (pred_map == class_idx))              # 计算并集长度
        source.append(intersection / union)

    return source


def LoadConfigFile(path='./config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def white_balance(img, mode=1, normal=True):
    """白平衡处理（默认为1均值、2完美反射、3灰度世界、4基于图像分析的偏色检测及颜色校正、5动态阈值）"""
    if normal:
        # 图像归一化回归
        img = (img * 255).astype(np.uint8)
        img[img > 255] = 255
        img[img < 0] = 0
    # 读取图像
    b, g, r = cv2.split(img)
    # 均值变为三通道
    h, w, c = img.shape
    if mode == 1:
        # 默认均值  ---- 简单的求均值白平衡法
        # 将计算三个通道的均值，然后将三个通道上的数值向均值拉近
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        # 求各个通道所占增益
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        output_img = cv2.merge([b, g, r])
    elif mode == 2:
        # 完美反射白平衡 ---- 依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
        # 找到图像中最亮的点，作为参照点，将参照点映射到255，然后根据映射关系去放大其他位置的像素情况
        output_img = img.copy()
        sum_ = np.double() + b + g + r
        hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
        Y = 765
        num, key = 0, 0
        ratio = 0.01
        while Y >= 0:
            num += hists[Y]
            if num > h * w * ratio / 100:
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(sum_ >= key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        times = len(sumkey[0])
        avg_b, avg_g, avg_r = sum_b / times, sum_g / times, sum_r / times

        maxvalue = float(np.max(output_img))
        output_img[:, :, 0] = output_img[:, :, 0] * maxvalue / int(avg_b)
        output_img[:, :, 1] = output_img[:, :, 1] * maxvalue / int(avg_g)
        output_img[:, :, 2] = output_img[:, :, 2] * maxvalue / int(avg_r)
    elif mode == 3:
        # 灰度世界假设
        # 假设认为一副有着大量彩色变化的图片，三个通道平均值应该趋向于同一个值k
        # 算法得到结果可能会存在溢出（>255，但不会<0）的情况
        # 处理方法：
        #   1、可以将其直接设置为255
        #   2、将整幅图映射回[0, 255]区间内
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        # 需要调整的RGB分量的增益
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        ba, ga, ra = b * kb, g * kg, r * kr

        output_img = cv2.merge([ba, ga, ra])
        output_img[output_img > 255] = 255
    elif mode == 4:
        # 基于图像分析的偏色检测及颜色校正
        I_b_2, I_r_2 = np.double(b) ** 2, np.double(r) ** 2
        sum_I_b_2, sum_I_r_2 = np.sum(I_b_2), np.sum(I_r_2)
        sum_I_b, sum_I_g, sum_I_r = np.sum(b), np.sum(g), np.sum(r)
        max_I_b, max_I_g, max_I_r = np.max(b), np.max(g), np.max(r)
        max_I_b_2, max_I_r_2 = np.max(I_b_2), np.max(I_r_2)
        [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
        [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
        b0 = np.uint8(u_b * (np.double(b) ** 2) + v_b * b)
        r0 = np.uint8(u_r * (np.double(r) ** 2) + v_r * r)
        output_img = cv2.merge([b0, g, r0])
    elif mode == 5:
        # 动态阈值算法 ---- 白点检测和白点调整
        # 只是白点检测不是与完美反射算法相同的认为最亮的点为白点，而是通过另外的规则确定
        def con_num(x):
            if x > 0:
                return 1
            if x < 0:
                return -1
            if x == 0:
                return 0

        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # YUV空间
        (y, u, v) = cv2.split(yuv_img)
        max_y = np.max(y.flatten())
        sum_u, sum_v = np.sum(u), np.sum(v)
        avl_u, avl_v = sum_u / (h * w), sum_v / (h * w)
        du, dv = np.sum(np.abs(u - avl_u)), np.sum(np.abs(v - avl_v))
        avl_du, avl_dv = du / (h * w), dv / (h * w)
        radio = 0.5  # 如果该值过大过小，色温向两极端发展

        # valuekey = np.where((np.abs(u - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du)
        #                     | (np.abs(v - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv))
        valuekey = np.where((np.abs(u - (1.5 * avl_u + avl_du)) < 1.5 * avl_du)
                            | (np.abs(v - (avl_v + avl_dv)) < 1.5 * avl_dv))
        num_y, yhistogram = np.zeros((h, w)), np.zeros(256)
        num_y[valuekey] = np.uint8(y[valuekey])
        yhistogram = np.bincount(np.uint8(num_y[valuekey].flatten()), minlength=256)
        ysum = len(valuekey[0])
        Y = 255
        num, key = 0, 0
        while Y >= 0:
            num += yhistogram[Y]
            if num > 0.1 * ysum:  # 取前10%的亮点为计算值，如果该值过大易过曝光，该值过小调整幅度小
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(num_y > key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        num_rgb = len(sumkey[0])

        b0 = np.double(b) * int(max_y) / (sum_b / num_rgb)
        g0 = np.double(g) * int(max_y) / (sum_g / num_rgb)
        r0 = np.double(r) * int(max_y) / (sum_r / num_rgb)

        output_img = cv2.merge([b0, g0, r0])
        output_img[output_img > 255] = 255
    else:
        raise TypeError('mode should be in [1,2,3,4,5]. Got {}'.format(mode))
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    return output_img
