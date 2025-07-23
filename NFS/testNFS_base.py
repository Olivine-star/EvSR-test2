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
# ✅ 路径读取函数（内联）=
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
    def __init__(self):
        self.lrList, self.hrList, self.path = [], [], []

        # 读取路径
        self.hrPath = paths.get('test_hr', '')
        self.lrPath = paths.get('test_lr', '')
        os.makedirs(savepath, exist_ok=True)          # 只建一次输出目录
        
        # 基本形状
        self.H, self.W, self.nTimeBins = 128, 128, 1500

        # ① 列出并排序全部 HR/LR 文件（假设都是 .npy）
        hr_files = sorted([f for f in os.listdir(self.hrPath) if f.endswith('.npy')])
        lr_files = sorted([f for f in os.listdir(self.lrPath) if f.endswith('.npy')])

        # ② 确保文件名一一对应
        assert hr_files == lr_files, "⚠️ HR 和 LR 文件名不一致，请检查！"

        # ③ 构建文件列表
        for fname in hr_files:
            self.hrList.append(os.path.join(self.hrPath, fname))
            self.lrList.append(os.path.join(self.lrPath, fname))
            self.path.append(fname)                   # 仅文件名，用于保存结果

        print(f"🔹 读取 {len(self.hrList)} 对 HR/LR 样本")

    def __getitem__(self, idx):
        # 读取事件 → spike tensor
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, self.H // 2, self.W // 2, self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)

        return eventLr1, eventHr1, self.path[idx]     # path 只是文件名

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

            eventList = getEventFromTensor(output)
            e = eventList[0]
            e = e[:, [0,2,1,3]]
            new_path = os.path.join(savepath, path[0])  # path[0] 现在就是 'xxx.npy'

            np.save(new_path, e.astype(np.int32))

        if k % 100 == 0:
            print(k, '/', len(testDataset))


if __name__ == '__main__':
    main()
