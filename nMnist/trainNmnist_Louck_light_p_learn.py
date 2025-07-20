import sys
import os
import datetime
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../')
from model_Louck_light import NetworkBasic
from nMnist.mnistDatasetSR import mnistDataset
from utils.ckpt import checkpoint_restore, checkpoint_save
from opts import parser
from statistic import Metric
import slayerSNN as snn
import numpy as np

from utils.drawloss import draw

import matplotlib.pyplot as plt
# from LOSS import ES1_loss
# from LOSS import ES1_loss_p

from LOSS.ES1_loss_p_learn import LearnableLoss

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

    loss_fn = LearnableLoss().to(device)


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


    
    MSE = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(
    list(m.parameters()) + list(loss_fn.parameters()), lr=args.lr, amsgrad=True)



    
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

    best_weights = {'epoch': -1, 'w1': None, 'w2': None, 'w3': None, 'loss': float('inf')}


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
            # loss_total, loss, loss_ecm = ES1_loss.training_loss(output, target, shape)
            # loss_total, loss, loss_ecm, loss_polarity = ES1_loss_p.training_loss(output, target, shape)
            loss_total, loss, loss_ecm, loss_polarity = loss_fn(output, target, shape)

            

            # 清空旧梯度。
            optimizer.zero_grad()
            # 反向传播计算新梯度。
            loss_total.backward()
            # 更新参数。
            optimizer.step()

            
            if i % showFreq == 0:
                trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_polarity.item(), loss_total.item(), 1,
                                       eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
                print_progress(epoch, maxEpoch, i, iter_per_epoch, bs, trainMetirc, time_last, "Train", log_training)
                time_last = datetime.datetime.now()    

        log_tensorboard(tf_writer, trainMetirc, epoch, prefix="Train")
        log_epoch_done(log_training, epoch)
        # ✅ 打印当前loss各项权重
        print("Loss Weights: w1=%.4f, w2=%.4f, w3=%.4f" % (
            torch.exp(-loss_fn.log_w1).item(),
            torch.exp(-loss_fn.log_w2).item(),
            torch.exp(-loss_fn.log_w3).item()
        ))


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


                    # loss_total, loss, loss_ecm, loss_polarity= ES1_loss_p.validation_loss(output, target, shape)
                    loss_total, loss, loss_ecm, loss_polarity = loss_fn(output, target, shape)

                    

                    valMetirc.updateIter(
                        loss.item(), loss_ecm.item(), loss_polarity.item(), loss_total.item(), 1,
                        eventLr.sum().item(), output.sum().item(), eventHr.sum().item()
                    )

                    if i % showFreq == 0:
                        print_progress(epoch, maxEpoch, i, len(testDataset) // bs, bs, valMetirc, time_last, "Val",
                                       log_training)
                        time_last = datetime.datetime.now()

            log_tensorboard(tf_writer, valMetirc, epoch, prefix="Val")
            log_validation_summary(valMetirc, valLossHistory, epoch, t, log_training, savePath, m, device, loss_fn, best_weights)

        # 学习率衰减（每15轮），将学习率缩小为原来的 0.1 倍，加快收敛。
        if (epoch + 1) % 8 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print("Learning rate decreased to:", param_group['lr'])

    print("\n Best validation loss: %.6f at epoch %d" % (best_weights['loss'], best_weights['epoch']))
    print("   Corresponding weights: w1=%.4f, w2=%.4f, w3=%.4f" % (
        best_weights['w1'], best_weights['w2'], best_weights['w3']
    ))

     #  保存为 txt 文件
    with open(os.path.join(savePath, 'best_loss_weights.txt'), 'w') as f:
        f.write(f"Best epoch: {best_weights['epoch']}\n")
        f.write(f"Best Val Loss: {best_weights['loss']:.6f}\n")
        f.write(f"Best w1: {best_weights['w1']:.6f}\n")
        f.write(f"Best w2: {best_weights['w2']:.6f}\n")
        f.write(f"Best w3: {best_weights['w3']:.6f}\n")
            

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

    avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()

    msg = f'{mode}, Cost {dt:.1f}s, Epoch[{epoch}], Iter {i}/{total}, Time Loss: {avgLossTime:.6f}, ' \
          f'Ecm Loss: {avgLossEcm:.6f}, Polarity Loss: {avgLossPolarity:.6f}, Avg Loss: {avgLoss:.6f}, ' \
          f'bs: {bs}, IS: {avgIS}, OS: {avgOS}, GS: {avgGS}, ' \
          f'Remain time: {int(h):02d}:{int(m):02d}:{int(s):02d}, End at: {end_time:%Y-%m-%d %H:%M:%S}'

    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


# 用于将训练或测试过程中的各种指标（如损失、输入和输出脉冲数量）记录到 TensorBoard 中，以便进行可视化分析。
def log_tensorboard(writer, metric, epoch, prefix="Train"):
    avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()

    writer.add_scalar(f'loss/{prefix}_Time_Loss', avgLossTime, epoch)
    writer.add_scalar(f'loss/{prefix}_Spatial_Loss', avgLossEcm, epoch)
    writer.add_scalar(f'loss/{prefix}_Polarity_Loss', avgLossPolarity, epoch)
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


# def log_validation_summary(metric, valLossHistory, epoch, t_start, log_file, savePath, model, device):
def log_validation_summary(metric, valLossHistory, epoch,  t_start, log_file, savePath, model, device, loss_fn, best_weights):
    avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, *_ = metric.getAvg()
    valLossHistory.append(avgLoss)
    t_end = datetime.datetime.now()
    
    msg = f"Validation Done! Cost Time: {(t_end - t_start).total_seconds():.2f}s, " \
          f"Loss Time: {avgLossTime:.6f}, Loss Ecm: {avgLossEcm:.6f}, Polarity Loss: {avgLossPolarity:.6f}, " \
          f"Avg Loss: {avgLoss:.6f}, Min Val Loss: {min(valLossHistory):.6f}\n"
    print(msg)
    
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

    checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckpt", device=device)

    if avgLoss < best_weights['loss']:
        best_weights['epoch'] = epoch
        best_weights['w1'] = torch.exp(-loss_fn.log_w1).item()
        best_weights['w2'] = torch.exp(-loss_fn.log_w2).item()
        best_weights['w3'] = torch.exp(-loss_fn.log_w3).item()
        best_weights['loss'] = avgLoss

    if avgLoss == min(valLossHistory):
        checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckptBest", device=device)

    with open(os.path.join(savePath, 'log.txt'), "a") as f:
        f.write(f"Epoch: {epoch}, Ecm loss: {avgLossEcm:.6f}, Spike time loss: {avgLossTime:.6f}, "
                f"Polarity loss: {avgLossPolarity:.6f}, Total loss: {avgLoss:.6f}\n")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    run()

