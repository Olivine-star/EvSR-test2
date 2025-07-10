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





def temporalSubsample(eventData, numSubstreams=5):
    """
    将事件数据按相等时间间隔分成多个子流
    
    Args:
        eventData: numpy数组，形状为(N, 4)，每行为[t, x, y, p]
        numSubstreams: 子流数量，默认为5
    
    Returns:
        list: 包含numSubstreams个子流的列表，每个子流都是numpy数组
    """
    if len(eventData) == 0:
        return [np.array([]).reshape(0, 4) for _ in range(numSubstreams)]
    
    # 获取时间范围
    minTime = eventData[:, 0].min()
    maxTime = eventData[:, 0].max()
    timeRange = maxTime - minTime
    
    
    
    # 如果时间范围为0，所有事件都在同一时刻
    if timeRange == 0:
        # 将所有事件分配给第一个子流，其他子流为空
        substreams = [eventData.copy()]
        for _ in range(numSubstreams - 1):
            substreams.append(np.array([]).reshape(0, 4))
        return substreams
    
    # 计算每个子流的时间间隔
    intervalSize = timeRange / numSubstreams
    substreams = []
    
    for i in range(numSubstreams):
        # 计算当前子流的时间边界
        startTime = minTime + i * intervalSize
        endTime = minTime + (i + 1) * intervalSize
        
        # 对于最后一个子流，包含最大时间点
        if i == numSubstreams - 1:
            mask = (eventData[:, 0] >= startTime) & (eventData[:, 0] <= endTime)
        else:
            mask = (eventData[:, 0] >= startTime) & (eventData[:, 0] < endTime)
        
        # 提取当前子流的事件
        substream = eventData[mask].copy()
        substreams.append(substream)
        
        
    
    return substreams





# 这个定义的类 mnistDataset 是一个继承自 Dataset 的自定义数据集类，用于处理和加载 MNIST 数据集的高分辨率（HR）和低分辨率（LR）数据。
class mnistDataset(Dataset):
    def __init__(self, train=True, path_config='dataset_path.txt'):
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

        # 将event对象转换为原始numpy形式 [t, x, y, p]，用于时间切分
        rawHr = np.stack([eventHr.t, eventHr.x, eventHr.y, eventHr.p], axis=-1)
        rawLr = np.stack([eventLr.t, eventLr.x, eventLr.y, eventLr.p], axis=-1)

        # 时间切分
        substreamsHr = temporalSubsample(rawHr, numSubstreams=5)
        substreamsLr = temporalSubsample(rawLr, numSubstreams=5)

        spikeHrList = []
        spikeLrList = []

        for i in range(5):
            # 转换为event对象
            subEvHr = event(substreamsHr[i][:,1], substreamsHr[i][:,2], substreamsHr[i][:,3], substreamsHr[i][:,0])
            subEvLr = event(substreamsLr[i][:,1], substreamsLr[i][:,2], substreamsLr[i][:,3], substreamsLr[i][:,0])

            # 转为spike张量
            spikeHr = subEvHr.toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)))
            spikeLr = subEvLr.toSpikeTensor(torch.zeros((2, 17, 17, self.nTimeBins)))

            spikeHrList.append(spikeHr)
            spikeLrList.append(spikeLr)

        # 将列表堆叠为5个batch的张量：(5, 2, H, W, T)
        spikeHrTensor = torch.stack(spikeHrList)
        spikeLrTensor = torch.stack(spikeLrList)

        return spikeLrTensor, spikeHrTensor


    def __len__(self):
        # 返回低分辨率事件列表的长度
        return len(self.lrList)

