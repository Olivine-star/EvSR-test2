import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    startTime = npEvent[:, 0].min()
    npEvent[:, 0] -= startTime
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3), startTime


class irDataset(Dataset):
    def __init__(self, train=True, path_config='../ir_path.txt'):
        self.lrList = []
        self.hrList = []

        self.H = 180
        self.W = 240
        self.nTimeBins = 50

        # 读取统一配置文件
        with open(path_config, 'r') as f:
            lines = f.read().splitlines()
            path_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines if '=' in line}

        if train:
            txt_path = path_dict.get('train_list', '')
        else:
            txt_path = path_dict.get('test_list', '')

        assert os.path.exists(txt_path), f"File not found: {txt_path}"

        with open(txt_path, 'r') as f:
            for line in f:
                lp, hp = line.strip().split()
                self.lrList.append(lp)
                self.hrList.append(hp)

    def __getitem__(self, idx):
        eventHr, startTime = readNpSpikes(self.hrList[idx])
        eventLr, startTime = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))
        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)
        return eventLr1, eventHr1, startTime

    def __len__(self):
        return len(self.lrList)
