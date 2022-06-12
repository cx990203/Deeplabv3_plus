from torch.utils.data.dataset import Dataset
import cv2
from PIL import Image
import numpy as np
from utils.utils import ImageResize


def normalize_input(image):
    image /= 255.0
    return image


class DeeplabDataloader(Dataset):

    def __init__(self, class_num: int, filenames: list, path: str, input_size: list):
        super(DeeplabDataloader, self).__init__()
        self.class_num: int = class_num
        self.filenames: list = filenames
        self.length = len(self.filenames)
        self.path = path
        self.input_size = input_size

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # 加载数据集
        img = Image.open(f"{self.path}/JPEGImages/{self.filenames[item]}.jpg")
        label = Image.open(f"{self.path}/SegmentationClassPNG/{self.filenames[item]}.png")
        # 图像转numpy
        img = np.array(img, np.float32)
        label = np.array(label)
        # resize
        img = ImageResize(img, self.input_size[0], self.input_size[1])
        label = ImageResize(label, self.input_size[0], self.input_size[1])
        # 图像通道转换，pytorch使用的是[b ,c, h, w]，并且进行像素归一化处理
        img = np.transpose(normalize_input(img), [2, 0, 1])
        return img, label
