ckpt_number=1
ckptPath="result/TestAutomatic"

import nMnist.trainNmnist_Louck as train
from types import SimpleNamespace
args = SimpleNamespace(
    bs=64,                         # batch size
    savepath=ckptPath,           # checkpoint 保存目录
    epoch=1,                      # 训练轮数
    showFreq=50,                  # 显示频率
    lr=0.1,                       # 学习率
    cuda='0',                     # 使用哪张 GPU
    j=4,                           # DataLoader 线程数（可选，也许 opts.py 有定义）
    dataset_path='dataset_path.txt',
    networkyaml='nMnist/network.yaml'
)
import torch.multiprocessing
torch.multiprocessing.freeze_support()
ckptPath = train.run(args)


import inference
savepath=ckptPath+"/inference"
#ckptPath="result/TestAutomatic/bs64_lr0.1_ep1_cuda0_20250714_000419"
inference.running_inference(ckpt_number=1, ckpt_root=ckptPath, hrPath=None, lrPath=None, base_savepath=savepath)