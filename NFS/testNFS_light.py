
import sys
sys.path.append('..')
from model_light import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import numpy as np
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore
from slayerSNN.spikeFileIO import event
import os
from utils.utils import getEventFromTensor

import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    startTime = npEvent[:, 0].min()
    npEvent[:, 0] -= startTime
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3), startTime






# -------------------------------
# ✅ 路径读取函数（内联）==
# -------------------------------
def load_path_config(path_config='../nfs_path.txt'):
    path_dict = {}
    with open(path_config, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                path_dict[key.strip()] = val.strip()
    return path_dict

# -------------------------------
# ✅ 加载路径配置
# -------------------------------
paths = load_path_config()
savepath = paths.get('savepath', '')
ckptPath = paths.get('ckptPath', '')



    

class nfsDataset(Dataset):
    def __init__(self):
        self.lrList = []
        self.hrList = []
        self.path = []

        self.H = 124
        self.W = 222
        self.nTimeBins = 50  # 时间维度固定

        hr_root = paths.get('test_hr')
        lr_root = paths.get('test_lr')
        os.makedirs(savepath, exist_ok=True)

        # ✅ 递归收集所有 .npy 文件路径（包括子文件夹）
        hr_files = sorted(glob.glob(os.path.join(hr_root, '*', '*.npy')))
        lr_files = sorted(glob.glob(os.path.join(lr_root, '*', '*.npy')))

        # ✅ 转为相对路径集合用于比对
        hr_set = set([os.path.relpath(p, hr_root) for p in hr_files])
        lr_set = set([os.path.relpath(p, lr_root) for p in lr_files])

        only_in_hr = sorted(hr_set - lr_set)
        only_in_lr = sorted(lr_set - hr_set)
        both = sorted(hr_set & lr_set)

        # 🔍 打印差异信息
        if only_in_hr:
            print(f"[❗] {len(only_in_hr)} 文件只存在于 HR 中：")
            for f in only_in_hr:
                print(f"  HR only: {f}")

        if only_in_lr:
            print(f"[❗] {len(only_in_lr)} 文件只存在于 LR 中：")
            for f in only_in_lr:
                print(f"  LR only: {f}")

        # ✅ 只加载两者都存在的文件
        for rel_path in both:
            self.hrList.append(os.path.join(hr_root, rel_path))
            self.lrList.append(os.path.join(lr_root, rel_path))
            self.path.append(rel_path)

        print(f"[✅] 加载配对样本数量: {len(self.hrList)}")











    def __getitem__(self, idx):
        eventHr,startTime = readNpSpikes(self.hrList[idx])
        eventLr,startTime = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        # assert eventHr1.sum() == len(eventHr.x)

       


        # assert eventLr1.sum() == len(eventLr.x)

        return eventLr1, eventHr1, startTime, self.path[idx]

    def __len__(self):
        return len(self.lrList)





def main():
    device = 'cuda'
    testDataset = nfsDataset()

    with open(os.path.join(savepath, 'ckpt.txt'), 'w') as f:
        f.writelines(ckptPath)

    bs = 1
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=1, drop_last=False)

    netParams = snn.params('network.yaml')
    m = NetworkBasic(netParams).to("cuda")
    m = torch.nn.DataParallel(m).to(device)
    m.eval()

    print(netParams['simulation'])

    m, epoch0 = checkpoint_restore(m, ckptPath, name="ckptBest")

    for k, (eventLr, eventHr, startTime, path) in enumerate(testLoader, 0):
        with torch.no_grad():
            eventLr = eventLr.to("cuda")
            eventHr = eventHr.to("cuda")

            output = m(eventLr)
            # eventHr = eventHr[:, :, :output.shape[2], :output.shape[3], :]

            eventList = getEventFromTensor(output)
            e = eventList[0]
            e = e[:, [0,2,1,3]]
            e[:, 0] = e[:, 0] + startTime[0].item()
            # new_path = os.path.join(savepath, path[0])  # path[0] 现在就是 'xxx.npy'
            new_path = os.path.join(savepath, path[0])
            os.makedirs(os.path.dirname(new_path), exist_ok=True)  # 创建目录
            np.save(new_path, e.astype(np.int32))







            np.save(new_path, e.astype(np.int32))

        if k % 100 == 0:
            print(k, '/', len(testDataset))


if __name__ == '__main__':
    main()
