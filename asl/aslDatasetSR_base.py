import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event
import random


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


class aslDataset(Dataset):
    def __init__(self, train=True, shape=[180, 240, 200], path_config='../asl_path.txt'):
        self.lrList = []
        self.hrList = []
        self.train = train
        self.H = shape[0]
        self.W = shape[1]

        classList = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

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

        for cls in classList:
            print("Read data", cls)
            hp = os.path.join(self.hrPath, cls)
            lp = os.path.join(self.lrPath, cls)
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            for n in os.listdir(hp):
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))


        self.nTimeBins = shape[2]

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        

        return eventLr1, eventHr1

    def __len__(self):
        # return 100
        return len(self.lrList)