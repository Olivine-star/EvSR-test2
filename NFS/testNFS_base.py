import sys
sys.path.append('..')
from model import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import numpy as np
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore
from slayerSNN.spikeFileIO import event
import os
from utils.utils import getEventFromTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


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




# ================================
# nfsDataset：去掉 classList 版本
# ================================
class nfsDataset(Dataset):
    def __init__(self, train=True, path_config='../nfs_path.txt'):
        self.lrList = []
        self.hrList = []
        self.samplingTime = 1.0  # ms

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

        # ★ 从第一个HR样本推断 H, W, nTimeBins
        first_event = np.load(self.hrList[0])
        self.W = int(first_event[:, 1].max()) + 1
        self.H = int(first_event[:, 2].max()) + 1
        t_max = first_event[:, 0].max() - first_event[:, 0].min()
        self.nTimeBins = int(np.ceil(t_max / self.samplingTime)) + 1

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(
            torch.zeros((2, int(self.H / 2), int(self.W / 2), self.nTimeBins)),
            samplingTime=self.samplingTime
        )
        eventHr1 = eventHr.toSpikeTensor(
            torch.zeros((2, self.H, self.W, self.nTimeBins)),
            samplingTime=self.samplingTime
        )

        return eventLr1, eventHr1, os.path.basename(self.hrList[idx])

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

    for k, (eventLr, eventHr, path) in enumerate(testLoader, 0):
        with torch.no_grad():
            eventLr = eventLr.to("cuda")
            eventHr = eventHr.to("cuda")

            output = m(eventLr)
            eventHr = eventHr[:, :, :output.shape[2], :output.shape[3], :]

            eventList = getEventFromTensor(output)
            e = eventList[0]
            e = e[:, [0,2,1,3]]
            new_path = os.path.join(savepath, path[0])  # path[0] 现在就是 'xxx.npy'

            np.save(new_path, e.astype(np.int32))

        if k % 100 == 0:
            print(k, '/', len(testDataset))


if __name__ == '__main__':
    main()
