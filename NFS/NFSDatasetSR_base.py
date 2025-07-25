import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event


# def readNpSpikes(filename):
#     npEvent = np.load(filename)
#     npEvent[:, 0] = npEvent[:, 0] - npEvent[:, 0].min()  # ← 归一化时间从0开始
#     return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * 1e-3)

def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    startTime = npEvent[:, 0].min()
    npEvent[:, 0] -= startTime
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3), startTime



class nfsDataset(Dataset):    
    def __init__(self, train=True, path_config='../nfs_path.txt'):
        self.lrList = []
        self.hrList = []
        self.train = train


        self.H = 124
        self.W = 222

        # self.H = 125
        # self.W = 223



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

    # def __getitem__(self, idx):
    #     eventHr = readNpSpikes(self.hrList[idx])
    #     eventLr = readNpSpikes(self.lrList[idx])

    #     eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H / 2), int(self.W / 2), self.nTimeBins)))
    #     eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

    #     return eventLr1, eventHr1





    def __getitem__(self, idx):
        eventHr, startTime = readNpSpikes(self.hrList[idx])
        eventLr, startTime = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))
        # assert eventHr1.sum() == len(eventHr.x)
        # assert eventLr1.sum() == len(eventLr.x)
        return eventLr1, eventHr1, startTime



    # def __getitem__(self, idx):
    #     eventHr, startTime = readNpSpikes(self.hrList[idx])
    #     eventLr, startTime = readNpSpikes(self.lrList[idx])

    #     # 转换为 Spike Tensor
    #     # eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        
    #     # 加1修正张量大小
    #     eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, (self.H+1)//2, (self.W+1)//2, self.nTimeBins)))

        
    #     eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

    #     # === ✅ 索引合法性检查（HR） ===
    #     assert eventHr.x.max() < self.W and eventHr.x.min() >= 0, \
    #         f"[HR] x out of range: min={eventHr.x.min()}, max={eventHr.x.max()}, expected [0, {self.W - 1}]"
    #     assert eventHr.y.max() < self.H and eventHr.y.min() >= 0, \
    #         f"[HR] y out of range: min={eventHr.y.min()}, max={eventHr.y.max()}, expected [0, {self.H - 1}]"
    #     assert eventHr.t.max() < self.nTimeBins and eventHr.t.min() >= 0, \
    #         f"[HR] t out of range: min={eventHr.t.min()}, max={eventHr.t.max()}, expected [0, {self.nTimeBins - 1}]"
    #     assert eventHr.p.max() <= 1 and eventHr.p.min() >= 0, \
    #         f"[HR] p out of range: unique={torch.unique(eventHr.p)}"

    #     # === ✅ 索引合法性检查（LR） ===
    #     assert eventLr.x.max() < self.W // 2 and eventLr.x.min() >= 0, \
    #         f"[LR] x out of range: min={eventLr.x.min()}, max={eventLr.x.max()}, expected [0, {self.W//2 - 1}]"
    #     assert eventLr.y.max() < self.H // 2 and eventLr.y.min() >= 0, \
    #         f"[LR] y out of range: min={eventLr.y.min()}, max={eventLr.y.max()}, expected [0, {self.H//2 - 1}]"
    #     assert eventLr.t.max() < self.nTimeBins and eventLr.t.min() >= 0, \
    #         f"[LR] t out of range: min={eventLr.t.min()}, max={eventLr.t.max()}, expected [0, {self.nTimeBins - 1}]"
    #     assert eventLr.p.max() <= 1 and eventLr.p.min() >= 0, \
    #         f"[LR] p out of range: unique={torch.unique(eventLr.p)}"

    #     # ✅ 数量对齐检查
    #     assert eventHr1.sum() == len(eventHr.x), f"[HR] spike tensor sum mismatch: expected {len(eventHr.x)}, got {eventHr1.sum()}"
    #     assert eventLr1.sum() == len(eventLr.x), f"[LR] spike tensor sum mismatch: expected {len(eventLr.x)}, got {eventLr1.sum()}"

    #     return eventLr1, eventHr1, startTime





    # def __getitem__(self, idx):
    #     eventHr, startTime = readNpSpikes(self.hrList[idx])
    #     eventLr, _ = readNpSpikes(self.lrList[idx])  # 同样归一化时间

    #     # === ✅ 计算 LR 尺寸 ===
    #     H_LR = (self.H + 1) // 2
    #     W_LR = (self.W + 1) // 2

    #     # === ✅ 构建 SpikeTensor 容器 ===
    #     eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, H_LR, W_LR, self.nTimeBins)))
    #     eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

    #     # === ✅ 合法性检查（HR）===
    #     assert eventHr.x.max() < self.W and eventHr.x.min() >= 0, \
    #         f"[HR] x out of range: min={eventHr.x.min()}, max={eventHr.x.max()}, expected [0, {self.W - 1}]"
    #     assert eventHr.y.max() < self.H and eventHr.y.min() >= 0, \
    #         f"[HR] y out of range: min={eventHr.y.min()}, max={eventHr.y.max()}, expected [0, {self.H - 1}]"
    #     assert eventHr.t.max() < self.nTimeBins and eventHr.t.min() >= 0, \
    #         f"[HR] t out of range: min={eventHr.t.min()}, max={eventHr.t.max()}, expected [0, {self.nTimeBins - 1}]"
    #     assert eventHr.p.max() <= 1 and eventHr.p.min() >= 0, \
    #         f"[HR] p out of range: unique={torch.unique(eventHr.p)}"

    #     # === ✅ 合法性检查（LR）===
    #     assert eventLr.x.max() < W_LR and eventLr.x.min() >= 0, \
    #         f"[LR] x out of range: min={eventLr.x.min()}, max={eventLr.x.max()}, expected [0, {W_LR - 1}]"
    #     assert eventLr.y.max() < H_LR and eventLr.y.min() >= 0, \
    #         f"[LR] y out of range: min={eventLr.y.min()}, max={eventLr.y.max()}, expected [0, {H_LR - 1}]"
    #     assert eventLr.t.max() < self.nTimeBins and eventLr.t.min() >= 0, \
    #         f"[LR] t out of range: min={eventLr.t.min()}, max={eventLr.t.max()}, expected [0, {self.nTimeBins - 1}]"
    #     assert eventLr.p.max() <= 1 and eventLr.p.min() >= 0, \
    #         f"[LR] p out of range: unique={torch.unique(eventLr.p)}"

    #     # === ✅ Spike Tensor 验证，容许微小丢失 ===
    #     hr_diff = abs(eventHr1.sum().item() - len(eventHr.x))
    #     lr_diff = abs(eventLr1.sum().item() - len(eventLr.x))

    #     assert hr_diff < 10, f"[HR] SpikeTensor sum mismatch: {eventHr1.sum()} vs {len(eventHr.x)} (diff={hr_diff})"
    #     assert lr_diff < 10, f"[LR] SpikeTensor sum mismatch: {eventLr1.sum()} vs {len(eventLr.x)} (diff={lr_diff})"


    #     return eventLr1, eventHr1, startTime



    def __len__(self):
        return len(self.lrList)
