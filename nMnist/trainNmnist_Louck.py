"""
这段代码是训练逻辑和打印结果
训练逻辑：
1.模型前向传播，输出结果。
output = m(eventLr)

2.计算损失函数。
其中一个loss是loss = MSE(output, eventHr)

3.反向传播，更新参数。
# 清空旧梯度。用optimizer的功能，优化器在110行定义：optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, amsgrad=True)
optimizer.zero_grad()
# 反向传播计算新梯度。通过先定义出来loss_total，用backward()计算梯度，再通过optimizer.step()更新参数。
loss_total.backward()
# 更新参数。
optimizer.step()

验证阶段：
验证阶段用于评估模型性能，不进行梯度更新和参数优化，仅前向传播并记录评估指标。

"""


import sys
import os
import datetime
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../')
from model_Louck_outchannel4 import NetworkBasic
from nMnist.mnistDatasetSR import mnistDataset
from utils.ckpt import checkpoint_restore, checkpoint_save
from opts import parser
from statistic import Metric
import slayerSNN as snn
import numpy as np

from utils.drawloss import draw

import matplotlib.pyplot as plt
from LOSS import ES1_loss

def run(args=None):
    if args is None:
        args = parser.parse_args()
    # 定义模型输入的形状
    shape = [34, 34, 350]
    # 设置环境变量，指定使用哪一块GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # 设置设备为GPU
    device = 'cuda'
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    # 创建训练数据集，读取训练数据集文件路径
    dataset_path = "../dataset_path.txt"
    if args.dataset_path is not None:
        dataset_path = args.dataset_path

    trainDataset = mnistDataset(path_config=dataset_path)
    # 创建测试数据集，读取训练测试集文件路径（False表明是测试集）
    testDataset = mnistDataset(False, dataset_path)

    print("Training sample: %d, Testing sample: %d" % (len(trainDataset), len(testDataset)))
    # 获取命令行参数（train.bat）中的batch size
    bs = args.bs

    # 使用 PyTorch 中的 DataLoader 来创建训练集数据加载器，batch_size为bs，
    # shuffle为True(表示在每个训练轮次开始前，随机打乱数据顺序，防止模型记住数据排列，从而提高泛化能力)，
    # num_workers为args.j，使用 j=4 个子进程（线程）来并行加载数据
    # drop_last为True,假设你训练集中有 10,000 个样本，batch size 是 64, 如果 drop_last=True，最后的 不足 64 个样本会被丢弃。
    # 如果 drop_last=False，最后的 batch 可能是 10000 % 64 = 16 个样本。
    # 训练时进行 梯度反向传播，每个 batch 要求 shape 一致,所以 drop_last=True；测试不需要梯度回传，对 batch 尺寸要求没那么严
    trainLoader = DataLoader(dataset=trainDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=False)

    # snn 是一个工具库，专门用于搭建和训练 SNN
    # 从 network.yaml 中读取 SNN 的仿真参数（如 Ts 时间步长、tSample 总时间窗），并返回一个参数字典或对象，供 NetworkBasic 初始化时使用。
    networkyaml = 'network.yaml'
    if args.networkyaml is not None:
        networkyaml = args.networkyaml
    netParams = snn.params(networkyaml)
    # 调用模型类，创建网络对象
    m = NetworkBasic(netParams)
    # 将网络转换为并行计算模式，并将其移动到指定的设备上
    m = torch.nn.DataParallel(m).to(device)

    # 打印网络
    print(m)


    # 定义均方误差损失函数
    MSE = torch.nn.MSELoss()
    # 定义Adam优化器，学习率为args.lr，amsgrad为True
    # “优化”就是通过反向传播计算梯度，并用优化器（如 Adam）根据这些梯度更新神经网络的参数，以使损失函数尽可能减小。
    # m.parameters() 表示模型 m 中所有需要训练的参数（如权重和偏置），是优化器更新的对象。
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, amsgrad=True)


    # 计算每个epoch的迭代次数。训练集样本总数➗bs，结果是 每个 epoch 需要几个 batch 才能跑完所有样本。如果训练集有 1000 个样本，bs = 100。就是说，每个 epoch 要迭代 10 次。
    # Batch size 是每次送入模型训练的一小批数据的数量。Epoch 是整个训练集被完整训练一遍的次数。
    iter_per_epoch = len(trainDataset) // bs
    # 获取当前时间
    time_last = datetime.datetime.now()

    # 创建保存文件的时间戳唯一标识文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建模型保存路径，根据命令行参数.bat中的savepath，文件名标识训练参数
    savePath = os.path.join(
        args.savepath,
        f"bs{args.bs}_lr{args.lr}_ep{args.epoch}_cuda{args.cuda}_{timestamp}"
    )

    # 创建保存路径，如果路径已存在则不报错（不会覆盖已有同名文件，如果目标文件夹已经存在，它就什么都不做。）
    os.makedirs(savePath, exist_ok=True)

    # 从savePath路径中恢复模型m和epoch0
    # m, epoch0 = checkpoint_restore(m, savePath)
    #用不同的 --savepath(train.bat改路径) 开启全新训练；如果以后这个文件夹里有 ckpt.pth，又能自动续训，两者兼容。
    ckpt_file = os.path.join(savePath, 'ckpt.pth')
    if os.path.exists(ckpt_file):
        m, epoch0 = checkpoint_restore(m, savePath)
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")
        epoch0 = -1  # 从头开始训练

    # 设置最大训练轮数
    maxEpoch = args.epoch
    # 设置显示频率
    showFreq = args.showFreq
    # 初始化验证损失历史记录
    valLossHistory = []
    # 创建TensorBoard写入器
    tf_writer = SummaryWriter(log_dir=savePath)

    # 打开保存路径下的config.txt文件，以写入模式打开
    with open(os.path.join(savePath, 'config.txt'), 'w') as f:
        # 遍历m.module.neuron_config中的每一个config
        for i, config in enumerate(m.module.neuron_config):
            # 将config中的参数写入文件
            f.writelines('layer%d: theta=%d, tauSr=%.2f, tauRef=%.2f, scaleRef=%.2f, tauRho=%.2f, scaleRho=%.2f\n' % (
                i + 1, config['theta'], config['tauSr'], config['tauRef'], config['scaleRef'], config['tauRho'], config['scaleRho']))
        # 写入一个空行
        f.writelines('\n')
        # 将args写入文件
        f.write(str(args))

    # 打开保存路径下的log.csv文件，以写入模式打开
    log_training = open(os.path.join(savePath, 'log.csv'), 'w')

############################################################ 模型主循环 #########################################################


    # 这段代码是你模型的训练主循环，每一轮（epoch）中都会进行训练和验证
    for epoch in range(epoch0 + 1, maxEpoch):
        trainMetirc = Metric()
        m.train()
        # 训练阶段（每 epoch）
        for i, (eventLr, eventHr) in enumerate(trainLoader, 0):
            eventLr, eventHr = eventLr.to(device), eventHr.to(device)  # [B, 2, H, W, T]

            # === 1. 拆分正负事件通道 ===
            eventLr_pos = eventLr[:, 0:1, ...]  # [B, 1, H, W, T]
            eventLr_neg = eventLr[:, 1:2, ...]  # [B, 1, H, W, T]
            eventHr_pos = eventHr[:, 0:1, ...]
            eventHr_neg = eventHr[:, 1:2, ...]


            # === 2. 前向传播两个通道分别 ===
            output_pos = m(eventLr_pos)  # [B, 1, H', W', T]
            output_neg = m(eventLr_neg)  # [B, 1, H', W', T]

            # === 3. 合并输出（拼回两个通道） ===
            output = torch.cat([output_pos, output_neg], dim=1)  # [B, 2, H', W', T]

            # === 4. 合并 GT ===
            target = torch.cat([eventHr_pos, eventHr_neg], dim=1)  # [B, 2, H', W', T]

            # === 5. 计算损失 ===
            loss_total, loss, loss_ecm = ES1_loss.training_loss(output, target, shape)
            """loss = MSE(output, target)
            # 时间块的 ECM loss
            loss_ecm = sum([
                MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
                    torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
                for i in range(shape[2] // 50)
            ])
            loss_total = loss + 5 * loss_ecm"""



            # 清空旧梯度。
            optimizer.zero_grad()
            # 反向传播计算新梯度。
            loss_total.backward()
            # 更新参数。
            optimizer.step()

            # 训练进度记录。每 showFreq 次迭代，记录一次当前指标，如损失、脉冲数量、预计剩余训练时间。
            if i % showFreq == 0:
                trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                       eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
                print_progress(epoch, maxEpoch, i, iter_per_epoch, bs, trainMetirc, time_last, "Train", log_training)
                time_last = datetime.datetime.now()

        log_tensorboard(tf_writer, trainMetirc, epoch, prefix="Train")
        log_epoch_done(log_training, epoch)

        # 验证阶段（每 epoch）
        if epoch % 1 == 0:
            m.eval()
            t = datetime.datetime.now()
            valMetirc = Metric()
            for i, (eventLr, eventHr) in enumerate(testLoader, 0):
                with torch.no_grad():
                    eventLr, eventHr = eventLr.to(device), eventHr.to(device)

                    # === 拆分正负事件 ===
                    eventLr_pos = eventLr[:, 0:1, ...]  # [B, 1, H, W, T]
                    eventLr_neg = eventLr[:, 1:2, ...]
                    eventHr_pos = eventHr[:, 0:1, ...]
                    eventHr_neg = eventHr[:, 1:2, ...]

                    # === 分别前向传播 ===
                    output_pos = m(eventLr_pos)
                    output_neg = m(eventLr_neg)

                    # === 合并输出 ===
                    output = torch.cat([output_pos, output_neg], dim=1)
                    target = torch.cat([eventHr_pos, eventHr_neg], dim=1)

                    # === 计算损失 ===
                    loss_total, loss, loss_ecm = ES1_loss.validation_loss(output, target, shape)
                    # loss = MSE(output, target)
                    # loss_ecm = sum([
                    #     MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
                    #         torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
                    #     for i in range(shape[2] // 50)
                    # ])
                    # loss_total = loss + loss_ecm

                    valMetirc.updateIter(
                        loss.item(), loss_ecm.item(), loss_total.item(), 1,
                        eventLr.sum().item(), output.sum().item(), eventHr.sum().item()
                    )

                    if i % showFreq == 0:
                        print_progress(epoch, maxEpoch, i, len(testDataset) // bs, bs, valMetirc, time_last, "Val",
                                       log_training)
                        time_last = datetime.datetime.now()

            log_tensorboard(tf_writer, valMetirc, epoch, prefix="Val")
            log_validation_summary(valMetirc, valLossHistory, epoch, t, log_training, savePath, m, device)

        # 学习率衰减（每15轮），将学习率缩小为原来的 0.1 倍，加快收敛。
        if (epoch + 1) % 15 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print("Learning rate decreased to:", param_group['lr'])

    return savePath


# 用于打印训练或测试过程中的进度信息
def print_progress(epoch, maxEpoch, i, total, bs, metric, time_last, mode, log_file):
    remainIter = (maxEpoch - epoch - 1) * total + (total - i - 1)
    now = datetime.datetime.now()
    dt = (now - time_last).total_seconds()
    remainSec = remainIter * dt
    h, remain = divmod(remainSec, 3600)
    m, s = divmod(remain, 60)
    end_time = now + datetime.timedelta(seconds=remainSec)
    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()
    msg = f'{mode}, Cost {dt:.1f}s, Epoch[{epoch}], Iter {i}/{total}, Time Loss: {avgLossTime:.6f}, ' \
          f'Ecm Loss: {avgLossEcm:.6f}, Avg Loss: {avgLoss:.6f}, bs: {bs}, IS: {avgIS}, OS: {avgOS}, GS: {avgGS}, ' \
          f'Remain time: {int(h):02d}:{int(m):02d}:{int(s):02d}, End at: {end_time:%Y-%m-%d %H:%M:%S}'
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

# 用于将训练或测试过程中的各种指标（如损失、输入和输出脉冲数量）记录到 TensorBoard 中，以便进行可视化分析。
def log_tensorboard(writer, metric, epoch, prefix="Train"):
    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()
    writer.add_scalar(f'loss/{prefix}_Time_Loss', avgLossTime, epoch)
    writer.add_scalar(f'loss/{prefix}_Spatial_Loss', avgLossEcm, epoch)
    writer.add_scalar(f'loss/{prefix}_Total_Loss', avgLoss, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_Input', avgIS, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_Output', avgOS, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_GT', avgGS, epoch)

# 定义一个函数，用于记录每个epoch完成的信息
def log_epoch_done(log_file, epoch):
    # 定义一个字符串，包含50个连字符，epoch的值，以及50个连字符
    msg = '-' * 50 + f"Epoch {epoch} Done" + '-' * 50
    # 打印该字符串
    print(msg)
    # 如果log_file不为空，则将字符串写入log_file，并刷新log_file
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

# 用于在验证过程中记录和保存模型的性能指标和检查点。它接受多个参数，包括验证指标、验证损失历史记录、当前 epoch、开始时间、日志文件、保存路径、模型和设备。
# 函数会计算平均损失和时间，打印和记录验证结果，并保存模型检查点。如果当前损失是最低的，还会保存一个最佳的检查点。
def log_validation_summary(metric, valLossHistory, epoch, t_start, log_file, savePath, model, device):
    avgLossTime, avgLossEcm, avgLoss, *_ = metric.getAvg()
    valLossHistory.append(avgLoss)
    t_end = datetime.datetime.now()
    msg = f"Validation Done! Cost Time: {(t_end - t_start).total_seconds():.2f}s, " \
          f"Loss Time: {avgLossTime:.6f}, Loss Ecm: {avgLossEcm:.6f}, Avg Loss: {avgLoss:.6f}, " \
          f"Min Val Loss: {min(valLossHistory):.6f}\n"
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

    # 保存模型
    checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckpt", device=device)
    # 如果平均损失等于验证损失历史中的最小值，则保存模型
    if avgLoss == min(valLossHistory):
        checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckptBest", device=device)
    # 打开日志文件，以追加方式写入
    with open(os.path.join(savePath, 'log.txt'), "a") as f:
        # 写入当前epoch的损失值
        f.write(f"Epoch: {epoch}, Ecm loss: {avgLossEcm:.6f}, Spike time loss: {avgLossTime:.6f}, Total loss: {avgLoss:.6f}\n")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    run()
