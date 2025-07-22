


"""
这一段代码的作用是：
1.输入eventLr，低分辨率事件
2.调用模型，对低分辨率事件eventLr进行预测，得到高分辨率事件保存为savepath（dataset_path.txt）下的.npy文件
"""


import sys
sys.path.append('..')
from model_Louck_light import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import datetime, os
import slayerSNN as snn
import torch
from utils.utils import getEventFromTensor
from utils.ckpt import checkpoint_restore
import numpy as np
from slayerSNN.spikeFileIO import event

import torch.jit



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)



class mnistDataset(Dataset):
    def __init__(self, hrPath, lrPath, savepath):
        self.lrList = []
        self.hrList = []
        # self.hrPath = paths.get('test_hr', '')
        # self.lrPath = paths.get('test_lr', '')
        self.hrPath = hrPath
        self.lrPath = lrPath
        self.path = []

        self.H = 34
        self.W = 34

        for k in range(10):
            #print("Read data %d"%k)
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



def inference(ckptPath, hrPath, lrPath, savepath, ckptname='ckptBest', networkyaml='nMnist/network.yaml'):
# 创建一个mnistDataset对象
    testDataset = mnistDataset(hrPath, lrPath, savepath)

    bs = 1
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=0)

    netParams = snn.params(networkyaml)
    m = NetworkBasic(netParams).to("cuda")
    m = torch.nn.DataParallel(m).to(device)
    m.eval()

    m, epoch0 = checkpoint_restore(m, ckptPath, name=ckptname, device=device)
    #print("start from epoch %d" % epoch0)
    stream_pos = torch.cuda.Stream(device=device)   # 正极事件
    stream_neg = torch.cuda.Stream(device=device)   # 负极事件




    import time
    total_time = 0.0
    log_file = open(os.path.join(savepath, 'inference_log.txt'), 'w', encoding='utf-8') 


    for k, (eventLr, eventHr, path) in enumerate(testLoader):
        with torch.no_grad():
            start_time = time.perf_counter()

            eventLr = eventLr.to("cuda")
            eventHr = eventHr.to("cuda")

             # ③ 拆分极性
            eventLr_pos = eventLr[:, 0:1, ...]
            eventLr_neg = eventLr[:, 1:2, ...]
            # eventHr_* 只在对比/可视化时用，这里可省

             # ④ 并发两个前向传播
            with torch.cuda.stream(stream_pos):
                output_pos = m(eventLr_pos)

            with torch.cuda.stream(stream_neg):
                output_neg = m(eventLr_neg)

            # ⑤ 主流等待两个子流完成
            torch.cuda.current_stream().wait_stream(stream_pos)
            torch.cuda.current_stream().wait_stream(stream_neg)

            # ⑥ 拼接结果并继续后处理
            output = torch.cat([output_pos, output_neg], dim=1)
            

            eventList = getEventFromTensor(output)
            e = eventList[0]
            e = e[:, [0, 2, 1, 3]]
            new_path = os.path.join(savepath, path[0])
            np.save(new_path, e.astype(np.int32))

            elapsed = time.perf_counter() - start_time
            total_time += elapsed

            if k % 100 == 0:
                msg = f"[{k}/{len(testLoader)}] Current sample inference time: {elapsed*1000:.2f} ms"
                print(msg)
                print(msg, file=log_file, flush=True)

    # 结束时打印统计
    msg1 = f"Total inference time: {total_time:.2f} seconds"
    msg2 = f"Average inference time per sample: {total_time / len(testLoader) * 1000:.2f} ms"
    print(msg1)
    print(msg2)
    print(msg1, file=log_file, flush=True)
    print(msg2, file=log_file, flush=True)

    log_file.close()


if __name__ == '__main__':

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
    hrPath = paths.get('test_hr', '')
    lrPath = paths.get('test_lr', '')

    inference(ckptPath, hrPath, lrPath, savepath, networkyaml='network.yaml')