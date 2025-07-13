"""
这一段代码的作用是：
1.输入eventLr，低分辨率事件
2.调用模型，对低分辨率事件eventLr进行预测，得到高分辨率事件保存为savepath（dataset_path.txt）下的.npy文件
"""


import sys
sys.path.append('..')
from 废弃.model_1 import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import os
import slayerSNN as snn
import torch
from utils.utils import getEventFromTensor
from utils.ckpt import checkpoint_restore
import numpy as np
from slayerSNN.spikeFileIO import event


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'


def readNpSpikes(filename, split_polarity=False, timeUnit=1e-3):
    npEvent = np.load(filename)

    if split_polarity:
        pos = npEvent[npEvent[:, 3] == 1]
        neg = npEvent[npEvent[:, 3] == 0]
        

        ev_pos = event(pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 0] * timeUnit * 1e3)
        ev_neg = event(neg[:, 1], neg[:, 2], neg[:, 3], neg[:, 0] * timeUnit * 1e3)




        return ev_pos, ev_neg  # 新增原始数组返回
    else:
        return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


# -------------------------------
# ✅ 路径读取函数（内联）
# -------------------------------
def load_path_config(path_config='dataset_path.txt'):
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
        

        # 读取高分辨率事件列表中的第idx个事件,readNpSpikes 函数用于读取这些文件，并将其转换为 event 对象
        # 👉 低分辨率事件：分开正负极性
        eventLr_pos, eventLr_neg = readNpSpikes(self.lrList[idx], split_polarity=True)

        # 👉 保证极性为通道 0
        eventLr_pos.p[:] = 0
        eventLr_neg.p[:] = 0

        # 👉 高分辨率事件：整体读取，不分极性
        eventHr = readNpSpikes(self.hrList[idx], split_polarity=False)

        # 转为 spike tensor（低分辨率两个极性通道）
        eventLr_pos_tensor = eventLr_pos.toSpikeTensor(torch.zeros((1, 17, 17, self.nTimeBins)))
        eventLr_neg_tensor = eventLr_neg.toSpikeTensor(torch.zeros((1, 17, 17, self.nTimeBins)))


        # 高分辨率事件直接转张量（默认含正负极性）
        eventHr_tensor = eventHr.toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)))

        # 校验


        assert eventLr_pos_tensor.sum() == len(eventLr_pos.x)
        assert eventLr_neg_tensor.sum() == len(eventLr_neg.x)
        
        assert eventHr_tensor.sum() == len(eventHr.x)

        path = self.path[idx]
        return eventLr_pos_tensor, eventLr_neg_tensor, eventHr_tensor, path

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

netParams = snn.params('network.yaml')
m = NetworkBasic(netParams).to("cuda")
m = torch.nn.DataParallel(m).to(device)
m.eval()

m, epoch0 = checkpoint_restore(m, ckptPath, name='ckptBest', device=device)
print("start from epoch %d" % epoch0)

Mse = torch.nn.MSELoss(reduction='mean')

loss_sum = 0
l = []
count = 0

lossTime = lossEcm = 0

for k, (eventLr_pos, eventLr_neg, eventHr, path) in enumerate(testLoader):
    with torch.no_grad():
        eventLr_pos = eventLr_pos.to("cuda")
        eventLr_neg = eventLr_neg.to("cuda")
        eventHr = eventHr.to("cuda")

        output_pos = m(eventLr_pos)
        output_neg = m(eventLr_neg)
        output = output_pos + output_neg 

        eventList = getEventFromTensor(output)
        e = eventList[0]
        e = e[:, [0, 2, 1, 3]]
        # 最后保存为 .npy 文件，输出路径为 savepath + 类别目录 + 文件名
        new_path = os.path.join(savepath, path[0])
        np.save(new_path, e.astype(np.int32))

        if k % 100 ==0:
            print("%d/%d"%(k, len(testLoader)))
