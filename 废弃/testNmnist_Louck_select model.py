"""
这一段代码的作用是：
1.输入eventLr，低分辨率事件
2.调用模型，对低分辨率事件eventLr进行预测，得到高分辨率事件保存为savepath（dataset_path.txt）下的.npy文件
"""


import sys
sys.path.append('..')
from model_Louck import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import os
import slayerSNN as snn
import torch
import numpy as np
from slayerSNN.spikeFileIO import event

from 废弃.select_model import run_multiple_models


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


# -------------------------------
# ✅ 路径读取函数（内联）
# -------------------------------
def load_path_config(path_config='../dataset_path.txt'):
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


class mnistDataset(Dataset):
    def __init__(self):
        self.lrList = []
        self.hrList = []
        self.hrPath = paths.get('test_hr', '')
        self.lrPath = paths.get('test_lr', '')
        self.path = []

        self.H = 34
        self.W = 34

        for k in range(10):
            print("Read data %d"%k)
            hp = os.path.join(self.hrPath, str(k))
            lp = os.path.join(self.lrPath, str(k))
            if not os.path.exists(os.path.join(savepath, str(k))):
                os.makedirs(os.path.join(savepath, str(k)))
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)
            for n in list:
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))
                self.path.append(os.path.join(str(k), n))

        self.nTimeBins = 350

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, 17, 17, self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)

        path = self.path[idx]
        return eventLr1, eventHr1, path

    def __len__(self):
        return len(self.lrList)


# 创建一个mnistDataset对象
testDataset = mnistDataset()
# 打开保存路径下的ckpt.txt文件，以写入模式
with open(os.path.join(savepath, 'ckpt.txt'), 'w') as f:
    # 将ckptPath写入文件
    f.writelines(ckptPath)

bs = 1
testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=0)

netParams = snn.params('../nMnist/network.yaml')

# m = NetworkBasic(netParams).to("cuda")
# m = torch.nn.DataParallel(m).to(device)
# m.eval()

# m, epoch0 = checkpoint_restore(m, ckptPath, name='ckptBest', device=device)
# print("start from epoch %d" % epoch0)

Mse = torch.nn.MSELoss(reduction='mean')

loss_sum = 0
l = []
count = 0

lossTime = lossEcm = 0



# ✅ 运行前10个模型，生成对应 HR 事件推理结果
run_multiple_models(
    model=NetworkBasic(netParams),
    testLoader=testLoader,
    base_ckpt_path=ckptPath,
    base_save_path=savepath,
    device=device
)
