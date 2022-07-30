import time
from typing import List

import numpy as np
from utils.utils import LoadConfigFile, IouSource, white_balance
from utils.dataloader import normalize_input
from model.deeplabv3 import Deeplabv3
import torch
from PIL import Image
import copy
from torch.nn import functional as F
from utils.utils import ImageResize
import os
from tqdm import tqdm
import cv2


def StyleChange(image: np.ndarray, change_mode=1) -> np.ndarray:
    if change_mode is None:
        return image

    if change_mode == 1:
        # 色调调整
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] = (image[:, :, 0] - 200) % 360
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 2:
        # 减小饱和度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 1] = image[:, :, 1] / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 3:
        # 减小亮度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 4:
        # 增大饱和度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 1] = image[:, :, 1] + (1 - image[:, :, 1]) / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 5:
        # 增大亮度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] + (1 - image[:, :, 2]) / 2.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    elif change_mode == 6:
        # 像素调整
        level = 10
        d = (1, 1, -1)      # 修正方向
        image[:, :, 0] = image[:, :, 0] + d[0] * level      # R
        image[:, :, 1] = image[:, :, 1] + d[1] * level      # G
        image[:, :, 2] = image[:, :, 2] - d[2] * level      # B
        image[image > 255] = 255
        image[image < 0] = 0

    return image


def main():
    # 加载配置文件
    config = LoadConfigFile()
    # 加载配置信息
    class_num = config['val']['num_class']
    input_size = config['val']['input_size']
    device = config['val']['device']
    para_path = config['val']['para_path']
    src_path = config['val']['src_path']
    label_path = config['val']['label_path']
    save_path = config['val']['save_path']
    save_flag = True        # 图像保存标志位
    show_flag = False       # 图像显示标志位
    tone_change_mode = 2
    white_balance_method: List[str] = ['none', 'mean', 'perfect_reflective', 'grey_world', 'image_analysis', 'dynamic_threshold']  # 所有可以使用的白平衡方法
    white_balance_method_using: str = ''  # 使用白平衡方法，如果不使用则为空''即可。只有开启了色调修改，白平衡方法才会生效
    if save_flag and (not os.path.exists(save_path)):
        os.mkdir(save_path)
    # 设置显示颜色
    color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # 设置运行设备
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置模型，并加载模型参数
    model = Deeplabv3(class_num=class_num, input_size=input_size)
    model.load_state_dict(torch.load(para_path, map_location=device))
    model.to(device)
    # 遍历源文件获取需要修改的图片目录
    files_list = []  # 需要修改的文件列表
    for root, dirs, files in os.walk(src_path, topdown=False):
        files_list = files
    # 开始进行模型验证
    model.eval()
    source_list = []
    with tqdm(total=len(files_list), desc=f'Validation') as pbar:
        for f in files_list:
            # 读取图片
            image = Image.open(f'{src_path}/{f}')
            image = np.array(image)                                         # 注意：这边读取完成的图片是BGR格式
            image = StyleChange(image, change_mode=tone_change_mode)        # 原图色调调整
            if white_balance_method_using in white_balance_method:
                # 图像白平衡处理
                mode = white_balance_method.index(white_balance_method_using)  # 选择白平衡模式
                image = white_balance(image, mode=mode, normal=False)  # 图像白平衡算法
            raw_image = image.copy()  # 拷贝原图
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            label = Image.open('{}/{}'.format(label_path, f.split('.')[0] + '.png'))
            label = np.array(label)
            label = ImageResize(label, input_size[0], input_size[1])
            label = torch.from_numpy(label)
            label = label.to(device)
            # Image -> numpy
            image = np.array(image, np.float32)
            image = ImageResize(image, input_size[0], input_size[1])
            image = np.transpose(normalize_input(image), [2, 0, 1])
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image)
            # 将输入转入device
            model.to(device)
            image = image.to(device)
            # 把图片放入模型当中进行计算
            pred = model(image)
            source_list.append(*IouSource(pred, label))
            # 结果转换成图片标注
            pred_temp = F.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous(), -1)
            pred_map = pred_temp.cpu().detach().numpy().argmax(axis=-1)[0, ...]
            # 将预测结果转换为图片
            seg_label = np.zeros([input_size[0], input_size[1], 3], dtype=np.int)
            for i in range(1, class_num):
                seg_label[:, :, 0] += (pred_map[:, :] == i) * np.array(color[i][0], dtype=np.int)
                seg_label[:, :, 1] += (pred_map[:, :] == i) * np.array(color[i][1], dtype=np.int)
                seg_label[:, :, 2] += (pred_map[:, :] == i) * np.array(color[i][2], dtype=np.int)
            seg_label = np.uint8(seg_label)
            # 图片混合
            res_image = cv2.addWeighted(raw_image, 0.5, seg_label, 0.5, 0)
            # 图片保存
            if save_flag:
                cv2.imwrite(f'{save_path}/{f}', res_image)
            # 显示图片
            if show_flag:
                cv2.namedWindow('show', cv2.WINDOW_NORMAL)
                cv2.imshow('show', np.hstack([raw_image, res_image]))
                cv2.waitKey(0)
            # 进度条更新
            pbar.set_postfix(**{
                'file name': f,
                'source-avg': sum(source_list) / len(source_list)
            })
            pbar.update(1)
    print('Validation finished!')
    print(f'Average source: \033[0;31m {sum(source_list) / len(source_list)} \033[0m')


if __name__ == '__main__':
    main()
