import sys
sys.path.append('../')
from model_Louck import NetworkBasic
from imageReconstruction.irDataset_base import irDataset
from torch.utils.data import DataLoader
import datetime
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore, checkpoint_save
import os
from opts import parser
from statistic import Metric
from tensorboardX import SummaryWriter

torch.backends.cudnn.enabled = False
import numpy as np

import matplotlib.pyplot as plt
from LOSS import ES1_loss

def main():

    args = parser.parse_args()

    shape = [180, 240, 50]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = 'cuda'
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)



    trainDataset = irDataset(train=True)
    testDataset = irDataset(train=False)
    print("Training sample: %d, Testing sample: %d" % (trainDataset.__len__(), testDataset.__len__()))
    bs = args.bs

    trainLoader = DataLoader(dataset=trainDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=False)

    netParams = snn.params('network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).to(device)
    print(m)

    MSE = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, amsgrad=True)

    iter_per_epoch = int(trainDataset.__len__() / bs)
    time_last = datetime.datetime.now()

    # savePath = args.savepath
    # savePath += "_bs%d" % args.bs
    # if args.add is not None:
    #     savePath += '_'
    #     savePath += args.add
    savePath = os.path.join(
                args.savepath,
                f"bs{args.bs}_lr{args.lr}_ep{args.epoch}_cuda{args.cuda}_{timestamp}"
            )

    os.makedirs(savePath, exist_ok=True)

    print(savePath)
    # m, epoch0 = checkpoint_restore(m, savePath)
    # 从savePath路径中恢复模型m和epoch0
    # m, epoch0 = checkpoint_restore(m, savePath)
    #用不同的 --savepath(train.bat改路径) 开启全新训练；如果以后这个文件夹里有 ckpt.pth，又能自动续训，两者兼容。
    ckpt_file = os.path.join(savePath, 'ckpt.pth')
    if os.path.exists(ckpt_file):
        m, epoch0 = checkpoint_restore(m, savePath)
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")
        epoch0 = -1  # 从头开始训练--



    maxEpoch = args.epoch
    showFreq = args.showFreq
    valLossHistory = []

    tf_writer = SummaryWriter(log_dir=savePath)
    with open(os.path.join(savePath, 'config.txt'), 'w') as f:
        for i, config in enumerate(m.module.neuron_config):
            f.writelines('layer%d: theta=%d, tauSr=%.2f, tauRef=%.2f, scaleRef=%.2f, tauRho=%.2f, scaleRho=%.2f\n' % (
                i + 1, config['theta'], config['tauSr'], config['tauRef'], config['scaleRef'], config['tauRho'],
                config['scaleRho']))
        f.writelines('\n')
        f.write(str(args))

    log_training = open(os.path.join(savePath, 'log.csv'), 'w')

    for epoch in range(epoch0+1, maxEpoch):
        trainMetirc = Metric()
        m.train()
        for i, (eventLr, eventHr, starttime) in enumerate(trainLoader, 0):

            num = eventLr.shape[0]
            eventLr = eventLr.to(device)
            eventHr = eventHr.to(device)



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
            

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if (i) % showFreq == 0:
                trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                    eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
                remainIter = (maxEpoch - epoch -1) * iter_per_epoch + (iter_per_epoch - i - 1)
                time_now = datetime.datetime.now()
                dt = (time_now - time_last).total_seconds()
                remainSec = remainIter * dt / showFreq
                minute, second = divmod(remainSec, 60)
                hour, minute = divmod(minute, 60)
                t1 = time_now + datetime.timedelta(seconds=remainSec)

                avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
                message = 'Train, Cost %.1fs, Epoch[%d]/[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Avg Loss: %f, ' \
                        'bs: %d, IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                        (dt, epoch, maxEpoch, i, iter_per_epoch, avgLossTime, avgLossEcm, avgLoss, bs, avgIS, avgOS, avgGS,
                        hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
                print(message)
                if log_training is not None:
                    log_training.write(message + '\n')
                    log_training.flush()
                time_last = time_now

        avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
        tf_writer.add_scalar('loss/Train_Time_Loss', avgLossTime, epoch)
        tf_writer.add_scalar('loss/Train_Spatial_Loss', avgLossEcm, epoch)
        tf_writer.add_scalar('loss/Train_Total_Loss', avgLoss, epoch)
        tf_writer.add_scalar('SpikeNum/Train_Input', avgIS, epoch)
        tf_writer.add_scalar('SpikeNum/Train_Output', avgOS, epoch)
        tf_writer.add_scalar('SpikeNum/Train_GT', avgGS, epoch)

        message = '-' * 50 + "Epoch %d Done" % epoch + '-' * 50
        print(message)
        if log_training is not None:
            log_training.write(message + '\n')
            log_training.flush()

        if epoch % 1 == 0:
            m.eval()
            t = datetime.datetime.now()
            valMetirc = Metric()
            for i, (eventLr, eventHr, starttime) in enumerate(testLoader, 0):
                with torch.no_grad():
                    num = eventLr.shape[0]
                    eventLr = eventLr.to(device)
                    eventHr = eventHr.to(device)



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



                    valMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                        eventLr.sum().item(), output.sum().item(), eventHr.sum().item())

                    if (i) % showFreq == 0:
                        remainIter = (maxEpoch - epoch - 1) * iter_per_epoch + (iter_per_epoch - i - 1)
                        time_now = datetime.datetime.now()
                        dt = (time_now - time_last).total_seconds()
                        remainSec = remainIter * dt / showFreq
                        minute, second = divmod(remainSec, 60)
                        hour, minute = divmod(minute, 60)
                        t1 = time_now + datetime.timedelta(seconds=remainSec)

                        avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
                        message = 'Val, Cost %.1fs, Epoch[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Avg Loss: %f,' \
                                ' IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                                (dt, epoch, i, len(testDataset)/args.bs, avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS,
                                hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
                        print(message)
                        if log_training is not None:
                            log_training.write(message + '\n')
                            log_training.flush()
                        time_last = time_now

            avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
            tf_writer.add_scalar('loss/Val_Time_Loss', avgLossTime, epoch)
            tf_writer.add_scalar('loss/Val_Spatial_Loss', avgLossEcm, epoch)
            tf_writer.add_scalar('loss/Val_Total_Loss', avgLoss, epoch)
            tf_writer.add_scalar('SpikeNum/Val_Input', avgIS, epoch)
            tf_writer.add_scalar('SpikeNum/Val_Output', avgOS, epoch)
            tf_writer.add_scalar('SpikeNum/Val_GT', avgGS, epoch)

            valLossHistory.append(avgLoss)
            time_last = datetime.datetime.now()
            message = "Validation Done! Cost Time: %.2fs, Loss Time: %f, Loss Ecm: %f, Avg Loss: %f, Min Val Loss: %f\n" %\
                    ((time_last-t).total_seconds(), avgLossTime, avgLossEcm, avgLoss, min(valLossHistory))
            print(message)
            if log_training is not None:
                log_training.write(message + '\n')
                log_training.flush()

            checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckpt", device=device)

            if (min(valLossHistory) == valLossHistory[-1]):
                checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckptBest", device=device)

            with open(os.path.join(savePath, 'log.txt'), "a") as f:
                f.write("Epoch: %d, Ecm loss: %f, Spike time loss: %f, Total loss: %f\n" %
                        (epoch, avgLossEcm, avgLossTime, avgLoss))

        if (epoch+1) % 15 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()