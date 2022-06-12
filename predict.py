import numpy as np
from utils.utils import LoadConfigFile
from utils.dataloader import normalize_input
from model.deeplabv3 import Deeplabv3
import torch
from PIL import Image
import copy
from torch.nn import functional as F
from utils.utils import ImageResize


def main():
    # 加载配置信息
    image_test = False
    config = LoadConfigFile()
    # 获取配置信息
    class_num = config['predict']['num_class']
    input_size = config['predict']['input_size']
    para_path = config['predict']['para_path']
    device = config['predict']['device']
    test_input = config['predict']['test_input']
    num_class = config['predict']['num_class']
    # 设置显示颜色
    color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # 设置运行设备
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置模型
    model = Deeplabv3(class_num=class_num, input_size=input_size)
    # 加载模型参数
    model.load_state_dict(torch.load(para_path, map_location=device))
    if '.jpg' in test_input:
        image_test = True
    model.eval()
    # 图片测试
    if image_test:
        # 打开测试文件
        image = Image.open(test_input)
        # 保留原图
        raw_image = copy.deepcopy(image)
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
        pred_temp = F.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous(), -1)
        pred_map = pred_temp.cpu().detach().numpy().argmax(axis=-1)[0, ...]
        # 将预测结果转换为图片
        seg_label = np.zeros([input_size[0], input_size[1], 3], dtype=np.int)
        for i in range(1, num_class):
            seg_label[:, :, 0] += (pred_map[:, :] == i) * np.array(color[i][0], dtype=np.int)
            seg_label[:, :, 1] += (pred_map[:, :] == i) * np.array(color[i][1], dtype=np.int)
            seg_label[:, :, 2] += (pred_map[:, :] == i) * np.array(color[i][2], dtype=np.int)
        seg_label = Image.fromarray(np.uint8(seg_label))
        # 图片混合
        res = Image.blend(raw_image, seg_label, 0.5)
        # 图片显示
        res.show()


if __name__ == '__main__':
    main()
