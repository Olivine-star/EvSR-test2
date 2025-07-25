import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event


def readNpSpikes(filename):
    npEvent = np.load(filename)
    npEvent[:, 0] = npEvent[:, 0] - npEvent[:, 0].min()  # ← 归一化时间从0开始
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * 1e-3)




class nfsDataset(Dataset):    
    def __init__(self, train=True, path_config='../nfs_path.txt'):
        self.lrList = []
        self.hrList = []
        self.train = train
        self.H = 125
        self.W = 223
        self.nTimeBins = 50

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

        # ✅ 遍历所有 HR 子文件夹和 .npy 文件
        for root, _, files in os.walk(self.hrPath):
            for file in files:
                if file.endswith('.npy'):
                    rel_path = os.path.relpath(os.path.join(root, file), self.hrPath)
                    hr_full = os.path.join(self.hrPath, rel_path)
                    lr_full = os.path.join(self.lrPath, rel_path)
                    if os.path.exists(lr_full):
                        self.hrList.append(hr_full)
                        self.lrList.append(lr_full)
                    else:
                        print(f"[警告] 跳过未匹配的样本: {rel_path}")


        print(f"[nfsDataset] Loaded {len(self.hrList)} samples from HR and LR.")

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H / 2), int(self.W / 2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        return eventLr1, eventHr1

    def __len__(self):
        return len(self.lrList)
