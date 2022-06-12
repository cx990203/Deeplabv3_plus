import torch.cuda
import numpy as np
from utils.dataloader import DeeplabDataloader
from model.deeplabv3 import Deeplabv3
from torch.utils.data.dataloader import DataLoader
from utils.Loss import CE_Loss
from utils.utils import weights_init, GetSource, LoadConfigFile
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import os
import time


def main():
    # 读取配置文件
    config = LoadConfigFile()
    # 获取配置信息
    class_num = config['train']['num_class']            # 类别总数
    datasets_path = config['train']['datasets_path']    # 数据集路径
    batch_size = config['train']['batch_size']          # 加载的batch size
    input_size = config['train']['input_size']          # 输入图像大小 [h, w]
    num_workers = config['train']['num_workers']        # 加载图像时使用的cpu核心数
    train_epoch = config['train']['train_epoch']        # 训练次数
    device = config['train']['device']                  # 使用训练设备，如果部位'cpu'，则默认加载gpu。暂时只支持单gpu使用
    para_path = config['train']['para_path']            # 与训练模型参数位置
    # 设置运行设备
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置类别权重
    class_weights = np.ones([class_num], np.float32)
    class_weights = torch.from_numpy(class_weights)
    # 创建网络
    model = Deeplabv3(class_num=class_num, input_size=input_size)
    # 初始化网络参数
    weights_init(model)
    if os.path.exists(para_path):
        model.load_state_dict(torch.load(para_path, map_location=device))
        print(f"Load model parameter from path: {para_path}")
    model.to(device)
    # 获取训练集和验证集文件名
    with open(f"{datasets_path}/ImageSets/train.txt", "r") as f:
        train_files = f.read().splitlines()
    with open(f"{datasets_path}/ImageSets/val.txt", "r") as f:
        val_files = f.read().splitlines()
    # 创建数据集
    train_dataset = DeeplabDataloader(class_num=class_num, filenames=train_files, path=datasets_path, input_size=input_size)
    val_dataset = DeeplabDataloader(class_num=class_num, filenames=val_files, path=datasets_path, input_size=input_size)
    # 加载数据集
    train_data = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_data = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    # 设置优化器
    opt = optim.SGD(model.parameters(), 4e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)

    # 训练前输出相关信息
    print(f"device: {device}")
    # 模型训练
    print("start training")
    # 设置一些运行相关参数
    valloss_best = 1000000
    root_path = './log'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    time_now = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    para_save_path = f'{root_path}/{time_now}'
    os.mkdir(para_save_path)

    # 开始训练
    train_loss = []     # 记录训练损失
    val_loss = []       # 记录验证损失
    cor = []            # 记录正确率
    for epoch in range(train_epoch):
        # 模型训练
        with tqdm(total=train_data.__len__(), desc=f'Train::Epoch {epoch + 1}/{train_epoch}') as pbar:
            # 设置模型为训练模式
            model.train()
            loss_avg = []
            for batchidx, batch in enumerate(train_data):
                # 导出数据
                x, label = batch
                x, label = x.to(device), label.to(device)
                class_weights = class_weights.to(device)
                # 前向计算
                pred = model(x)
                # 计算损失
                loss = CE_Loss(pred, label, class_weights, num_classes=class_num)
                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()
                # 记录损失数值
                loss_avg.append(loss.item())
                # 进度条更新
                pbar.set_postfix(**{
                    'loss': loss.item()
                })
                pbar.update(1)
        print(f'train::epoch: {epoch + 1}  avg loss: {sum(loss_avg) / len(loss_avg)}')
        train_loss.append(sum(loss_avg) / len(loss_avg))
        # 模型验证
        with tqdm(total=val_data.__len__(), desc=f'Val::Epoch {epoch + 1}/{train_epoch}') as pbar:
            # 设置模型为验证模式
            model.eval()
            source_avg = []
            loss_avg = []
            for batchidx, batch in enumerate(train_data):
                # 导出数据
                x, label = batch
                x, label = x.to(device), label.to(device)
                # 前向计算
                pred = model(x)
                # 计算损失
                loss = CE_Loss(pred, label, class_weights, num_classes=class_num)
                loss_avg.append(loss.item())
                # 判断是否为最佳参数，如果是则进行参数保存
                if loss.item() < valloss_best:
                    valloss_best = loss.item()
                    torch.save(model.state_dict(), f'{para_save_path}/best.pth')
                # 计算得分
                source_avg.append(GetSource(pred, label))
                # 进度条更新
                pbar.set_postfix(**{
                    'loss': loss.item(),
                    'cor': source_avg[-1]
                })
                pbar.update(1)
            print(f'val::epoch: {epoch + 1}  avg loss: {sum(loss_avg) / len(loss_avg)}  cor: {sum(source_avg) / len(source_avg)}')
            val_loss.append(sum(loss_avg) / len(loss_avg))
            cor.append(sum(source_avg) / len(source_avg))
        # 绘制损失曲线图像并保存
        plt.figure()
        plt.plot(range(epoch + 1), train_loss, 'r--', label='train loss')        # 绘制训练损失
        plt.plot(range(epoch + 1), val_loss, 'b--', label='val loss')            # 绘制验证损失
        plt.plot(range(epoch + 1), cor, 'g--', label='cor')                      # 绘制正确率
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'{para_save_path}/loss.png', dpi=300)


if __name__ == '__main__':
    main()
