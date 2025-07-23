import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event



def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

class nfsDataset(Dataset):
    def __init__(self, train=True, path_config='../nfs_path.txt'):
        self.lrList = []
        self.hrList = []
        self.H = 128
        self.W = 128
        self.nTimeBins = 1500

        # 读取路径配置文件
        with open(path_config, 'r') as f:
            lines = f.read().splitlines()
            path_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines if '=' in line}

        if train:
            self.hrPath = path_dict.get('train_hr', '')
            self.lrPath = path_dict.get('train_lr', '')
        else:
            self.hrPath = path_dict.get('test_hr', '')
            self.lrPath = path_dict.get('test_lr', '')

        # 获取所有文件名（假设HR和LR命名完全对应）
        hr_files = sorted(os.listdir(self.hrPath))
        lr_files = sorted(os.listdir(self.lrPath))

        assert len(hr_files) == len(lr_files), "HR and LR file counts do not match."

        for hr_file, lr_file in zip(hr_files, lr_files):
            self.hrList.append(os.path.join(self.hrPath, hr_file))
            self.lrList.append(os.path.join(self.lrPath, lr_file))

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H / 2), int(self.W / 2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)

        return eventLr1, eventHr1

    def __len__(self):
        return len(self.lrList)
