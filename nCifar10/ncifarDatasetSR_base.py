import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event

# classList = ['frog','dog','deer','ship','bird','horse','airplane','automobile','cat','truck']

classList = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


class ncifarDataset(Dataset):
    def __init__(self, train=True, shape=[128, 128, 1500], path_config='dataset_cifar.txt'):
        self.lrList = []
        self.hrList = []
        self.H = shape[1]
        self.W = shape[0]
        self.nTimeBins = shape[2]
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

        for k in range(10):
            print("Read data " + classList[k])
            hp = os.path.join(self.hrPath, classList[k])
            lp = os.path.join(self.lrPath, classList[k])
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)
            for n in list:
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H / 2), int(self.W / 2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)

        return eventLr1, eventHr1

    def __len__(self):
        # return 5
        return len(self.lrList)

