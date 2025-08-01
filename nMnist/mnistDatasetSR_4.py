"""
读取.npy文件，并返回一个event对象，并转化成张量

event.toSpikeTensor(torch.zeros((2, H, W, T)))把连续的异步事件流转换成一个固定形状的四维张量
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event

# 这个函数的作用是读取一个 .npy 文件，并返回一个 event 对象，其中时间数据被转换为毫秒
def readNpSpikes(filename, timeUnit=1e-3):
    # 读取文件名为filename的npy文件
    npEvent = np.load(filename)
    # 返回一个event对象，它接受四个参数，参数为npEvent的列1、列2、列3和列0乘以timeUnit再乘以1000
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

# 这个定义的类 mnistDataset 是一个继承自 Dataset 的自定义数据集类，用于处理和加载 MNIST 数据集的高分辨率（HR）和低分辨率（LR）数据。
class mnistDataset(Dataset):
    def __init__(self, train=True, path_config='../dataset_path.txt'):
        self.lrList = []
        self.hrList = []

        # 读取路径配置文件
        with open(path_config, 'r') as f:
            lines = f.read().splitlines()
            path_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines if '=' in line}

        if train:
            # 如果是训练集，则读取训练集的高分辨率图像路径和低分辨率图像路径
            self.hrPath = path_dict.get('train_hr', '')
            self.lrPath = path_dict.get('train_lr', '')
        else:
            # 如果是测试集，则读取测试集的高分辨率图像路径和低分辨率图像路径
            self.hrPath = path_dict.get('test_hr', '')
            self.lrPath = path_dict.get('test_lr', '')

        # 设置高分辨率图像和低分辨率图像的尺寸
        self.H = 34
        self.W = 34

        # 循环10次,对应MNIST数据集中的10个数字类别
        for k in range(10):
            # 打印读取数据
            print("Read data %d"%k)
            # 获取高分辨率图片路径
            hp = os.path.join(self.hrPath, str(k))
            # 获取低分辨率图片路径
            lp = os.path.join(self.lrPath, str(k))
            # 断言高分辨率图片数量和低分辨率图片数量相等
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            # 获取高分辨率图片列表
            list = os.listdir(hp)

            # 循环遍历高分辨率图片列表
            for n in list:
                # 将高分辨率事件的路径添加到hrList中,hrList 是一个包含高分辨率事件文件路径的列表，用于存储高分辨率事件文件路径。
                self.hrList.append(os.path.join(hp, n))
                # 将低分辨率图片路径添加到lrList中
                self.lrList.append(os.path.join(lp, n))

        # 设置时间间隔为350
        self.nTimeBins = 350

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        # 🔽 将 LR 的空间分辨率从 17x17 映射到 8x8
        eventLr.x = (eventLr.x // 2).astype(int)  # 原来是0~16 -> 现在是0~8
        eventLr.y = (eventLr.y // 2).astype(int)

        # ✅ 转换成 spike tensor
        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/4), int(self.W/4), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        # assert eventHr1.sum() == len(eventHr.x)
        # assert eventLr1.sum() == len(eventLr.x)

        return eventLr1, eventHr1


    def __len__(self):
        # 返回低分辨率事件列表的长度
        return len(self.lrList)

