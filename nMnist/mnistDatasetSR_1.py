"""
读取.npy文件，并返回一个event对象，并转化成张量

event.toSpikeTensor(torch.zeros((2, H, W, T)))把连续的异步事件流转换成一个固定形状的四维张量
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event

# # 这个函数的作用是读取一个 .npy 文件，并返回一个 event 对象，其中时间数据被转换为毫秒
# def readNpSpikes(filename, split_polarity=False, timeUnit=1e-3):
#     npEvent = np.load(filename)

#     if split_polarity:
#         # 分离正负极性
#         pos = npEvent[npEvent[:, 3] == 1]
#         neg = npEvent[npEvent[:, 3] == 0]

#         ev_pos = event(pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 0] * timeUnit * 1e3)
#         ev_neg = event(neg[:, 1], neg[:, 2], neg[:, 3], neg[:, 0] * timeUnit * 1e3)

#         return ev_pos, ev_neg
#     else:
#         # 返回整体事件（不区分极性）
#         return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)






# def readNpSpikes(filename, split_polarity=False, timeUnit=1e-3):
#     npEvent = np.load(filename)

#     if split_polarity:
#         pos = npEvent[npEvent[:, 3] == 1]
#         neg = npEvent[npEvent[:, 3] == 0]
#         print("分离正负极性的原始数据如下:")
#         print(pos)
#         print("===============================")
#         print(neg)

#         ev_pos = event(pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 0] * timeUnit * 1e3)
#         ev_neg = event(neg[:, 1], neg[:, 2], neg[:, 3], neg[:, 0] * timeUnit * 1e3)

#         print("===============================")
#         print("===============================")
#         print("===============================")


#         print("转换为事件对象后的数据如下:")
#         # print(ev_pos.x, ev_pos.y, ev_pos.p, ev_pos.t)
#         # print("===============================")
#         # print(ev_neg.x, ev_neg.y, ev_neg.p, ev_neg.t)

#         print(ev_pos.p)
#         print("===============================")
#         print(ev_neg.p)



#         return ev_pos, ev_neg, pos, neg   # 新增原始数组返回
#     else:
#         return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)





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








# 这个定义的类 mnistDataset 是一个继承自 Dataset 的自定义数据集类，用于处理和加载 MNIST 数据集的高分辨率（HR）和低分辨率（LR）数据。
class mnistDataset(Dataset):
    def __init__(self, train=True, path_config='dataset_path.txt'):
        self.lrList = []
        self.hrList = []

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

        # 设置高分辨率图像和低分辨率图像的尺寸
        self.H = 34
        self.W = 34

        # 循环10次,对应MNIST数据集中的10个数字类别
        for k in range(10):
            # 打印读取数据
            print("Read data %d"%k)
            # 获取高分辨率图片路径
            hp = os.path.join(self.hrPath, str(k))
            # 获取低分辨率图片路径
            lp = os.path.join(self.lrPath, str(k))
            # 断言高分辨率图片数量和低分辨率图片数量相等
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            # 获取高分辨率图片列表
            list = os.listdir(hp)

            # 循环遍历高分辨率图片列表
            for n in list:
                # 将高分辨率事件的路径添加到hrList中,hrList 是一个包含高分辨率事件文件路径的列表，用于存储高分辨率事件文件路径。
                self.hrList.append(os.path.join(hp, n))
                # 将低分辨率图片路径添加到lrList中
                self.lrList.append(os.path.join(lp, n))

        # 设置时间间隔为350
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

        # 返回：低分辨率张量（正负分通道），高分辨率张量（合在一起）
        return eventLr_pos_tensor, eventLr_neg_tensor, eventHr_tensor

    def __len__(self):
        # 返回低分辨率事件列表的长度
        return len(self.lrList)



def main():
    file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\LR\0\0.npy"
    # 载入原始事件
    all_events = np.load(file_path)
    print("事件总数:", all_events.shape[0])
    print("极性列唯一值:", np.unique(all_events[:, 3]))


    # 多接收两个原始 numpy 数组
    event_pos, event_neg, pos_raw, neg_raw = readNpSpikes(file_path, split_polarity=True)

    print("✅ 正极性事件数量:", len(pos_raw))
    print("✅ 负极性事件数量:", len(neg_raw))

    print("\n📘 正极性事件（前5条 原始数据）:")
    print(pos_raw[:5])  # 正确显示 [t, x, y, p=1]

    print("\n📕 负极性事件（前5条 原始数据）:")
    print(neg_raw[:5])  # 正确显示 [t, x, y, p=0]





if __name__ == "__main__":
    main()
