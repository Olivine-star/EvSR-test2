import sys
sys.path.append('../')
from model_Louck_light import NetworkBasic
from NFS.NFSDatasetSR_base import nfsDataset
from torch.utils.data import DataLoader
import datetime
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore, checkpoint_save
import os
from opts import parser
from statistic import Metric
from tensorboardX import SummaryWriter
import numpy as np

torch.backends.cudnn.enabled = False

import matplotlib.pyplot as plt
# from LOSS import ES1_loss
# from LOSS import ES1_loss_p
from LOSS.ES1_loss_p_learn import LearnableLoss

def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = 'cuda'
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    loss_fn = LearnableLoss().to(device)

    # shape = [224, 126, 1500]
    # trainDataset = nfsDataset(train=True)
    # testDataset = nfsDataset(train=False)

    # ① 实例化数据集之后，直接读取真实 shape=
    trainDataset = nfsDataset(train=True)
    testDataset  = nfsDataset(train=False)

    shape = [trainDataset.H, trainDataset.W, trainDataset.nTimeBins]   # ← 替换原来的手写 [224,126,1500]





    print("Training sample: %d, Testing sample: %d" % (trainDataset.__len__(), testDataset.__len__()))
    bs = args.bs

    trainLoader = DataLoader(dataset=trainDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=False)

    netParams = snn.params('network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).to(device)
    print(m)

    MSE = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(
    list(m.parameters()) + list(loss_fn.parameters()), lr=args.lr, amsgrad=True)


    iter_per_epoch = int(trainDataset.__len__() / bs)
    time_last = datetime.datetime.now()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # savePath = args.savepath
    savePath = os.path.join(
            args.savepath,
            f"bs{args.bs}_lr{args.lr}_ep{args.epoch}_cuda{args.cuda}_{timestamp}"
        )

    # 创建保存路径，如果路径已存在则不报错（不会覆盖已有同名文件，如果目标文件夹已经存在，它就什么都不做。）=
    os.makedirs(savePath, exist_ok=True)

    print(savePath)
    # m, epoch0 = checkpoint_restore(m, savePath, name='ckpt')=
    # 从savePath路径中恢复模型m和epoch0
    # m, epoch0 = checkpoint_restore(m, savePath)
    #用不同的 --savepath(train.bat改路径) 开启全新训练；如果以后这个文件夹里有 ckpt.pth，又能自动续训，两者兼容。
    ckpt_file = os.path.join(savePath, 'ckpt.pth')
    if os.path.exists(ckpt_file):
        m, epoch0 = checkpoint_restore(m, savePath)
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")
        epoch0 = -1  # 从头开始训练



    maxEpoch = args.epoch
    showFreq = args.showFreq
    valLossHistory = []


    best_weights = {'epoch': -1, 'w1': None, 'w2': None, 'w3': None, 'loss': float('inf')}


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
        # for i, (eventLr, eventHr) in enumerate(trainLoader, 0):
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

            loss_total, loss, loss_ecm, loss_polarity = loss_fn(output, target, shape)



            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if i % showFreq == 0:
                trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_polarity.item(), loss_total.item(), 1,
                                       eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
                
                remainIter = (maxEpoch - epoch -1) * iter_per_epoch + (iter_per_epoch - i - 1)
                time_now = datetime.datetime.now()
                dt = (time_now - time_last).total_seconds()
                remainSec = remainIter * dt / showFreq
                minute, second = divmod(remainSec, 60)
                hour, minute = divmod(minute, 60)
                t1 = time_now + datetime.timedelta(seconds=remainSec)

                avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
                
                message = 'Train, Cost %.1fs, Epoch[%d]/[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Polarity Loss: %f, Avg Loss: %f, ' \
                        'bs: %d, IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                        (dt, epoch, maxEpoch, i, iter_per_epoch, avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, bs, avgIS, avgOS, avgGS,
                        hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
                print(message)
                if log_training is not None:
                    log_training.write(message + '\n')
                    log_training.flush()
                time_last = time_now

        avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
        tf_writer.add_scalar('loss/Train_Time_Loss', avgLossTime, epoch)
        tf_writer.add_scalar('loss/Train_Spatial_Loss', avgLossEcm, epoch)
        tf_writer.add_scalar('loss/Train_Polarity_Loss', avgLossPolarity, epoch)
        tf_writer.add_scalar('loss/Train_Total_Loss', avgLoss, epoch)
        tf_writer.add_scalar('SpikeNum/Train_Input', avgIS, epoch)
        tf_writer.add_scalar('SpikeNum/Train_Output', avgOS, epoch)
        tf_writer.add_scalar('SpikeNum/Train_GT', avgGS, epoch)

        message = '-' * 50 + "Epoch %d Done" % epoch + '-' * 50
        print(message)
        if log_training is not None:
            log_training.write(message + '\n')
            log_training.flush()
        
        # ✅ 打印当前loss各项权重
        print("Loss Weights: w1=%.4f, w2=%.4f, w3=%.4f" % (
            torch.exp(-loss_fn.log_w1).item(),
            torch.exp(-loss_fn.log_w2).item(),
            torch.exp(-loss_fn.log_w3).item()
        ))

        if epoch % 1 == 0:
            m.eval()
            t = datetime.datetime.now()
            valMetirc = Metric()
            # for i, (eventLr, eventHr) in enumerate(testLoader, 0):

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

                    loss_total, loss, loss_ecm, loss_polarity = loss_fn(output, target, shape)


                    valMetirc.updateIter(loss.item(), loss_ecm.item(), loss_polarity.item(), loss_total.item(), 1,
                                        eventLr.sum().item(), output.sum().item(), eventHr.sum().item())


                    if (i) % showFreq == 0:
                        remainIter = (maxEpoch - epoch - 1) * iter_per_epoch + (iter_per_epoch - i - 1)
                        time_now = datetime.datetime.now()
                        dt = (time_now - time_last).total_seconds()
                        remainSec = remainIter * dt / showFreq
                        minute, second = divmod(remainSec, 60)
                        hour, minute = divmod(minute, 60)
                        t1 = time_now + datetime.timedelta(seconds=remainSec)

                        avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
                        message = 'Val, Cost %.1fs, Epoch[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Polarity Loss: %f, Avg Loss: %f,' \
                                ' IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                                (dt, epoch, i, len(testDataset)/args.bs, avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS,
                                hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
                        
                        print(message)
                        if log_training is not None:
                            log_training.write(message + '\n')
                            log_training.flush()
                        time_last = time_now

            avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
            tf_writer.add_scalar('loss/Val_Time_Loss', avgLossTime, epoch)
            tf_writer.add_scalar('loss/Val_Spatial_Loss', avgLossEcm, epoch)
            tf_writer.add_scalar('loss/Val_Polarity_Loss', avgLossPolarity, epoch)
            tf_writer.add_scalar('loss/Val_Total_Loss', avgLoss, epoch)
            tf_writer.add_scalar('SpikeNum/Val_Input', avgIS, epoch)
            tf_writer.add_scalar('SpikeNum/Val_Output', avgOS, epoch)
            tf_writer.add_scalar('SpikeNum/Val_GT', avgGS, epoch)

            valLossHistory.append(avgLoss)
            time_last = datetime.datetime.now()
            message = "Validation Done! Cost Time: %.2fs, Loss Time: %f, Loss Ecm: %f, Polarity Loss: %f, Avg Loss: %f, Min Val Loss: %f\n" %\
                    ((time_last-t).total_seconds(), avgLossTime, avgLossEcm, avgLossPolarity, avgLoss, min(valLossHistory))
            if log_training is not None:
                log_training.write(message + '\n')
                log_training.flush()

            checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckpt", device=device)


            if avgLoss < best_weights['loss']:
                best_weights['epoch'] = epoch
                best_weights['w1'] = torch.exp(-loss_fn.log_w1).item()
                best_weights['w2'] = torch.exp(-loss_fn.log_w2).item()
                best_weights['w3'] = torch.exp(-loss_fn.log_w3).item()
                best_weights['loss'] = avgLoss





            if (min(valLossHistory) == valLossHistory[-1]):
                checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckptBest", device=device)

            with open(os.path.join(savePath, 'log.txt'), "a") as f:
                f.write("Epoch: %d, Ecm loss: %f, Spike time loss: %f, Polarity loss: %f, Total loss: %f\n" %
                        (epoch, avgLossEcm, avgLossTime, avgLossPolarity, avgLoss))

        if (epoch+1) % 15 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])
    
    print("\n Best validation loss: %.6f at epoch %d" % (best_weights['loss'], best_weights['epoch']))
    print("   Corresponding weights: w1=%.4f, w2=%.4f, w3=%.4f" % (
        best_weights['w1'], best_weights['w2'], best_weights['w3']
    ))

     #  保存为 txt 文件=
    with open(os.path.join(savePath, 'best_loss_weights.txt'), 'w') as f:
        f.write(f"Best epoch: {best_weights['epoch']}\n")
        f.write(f"Best Val Loss: {best_weights['loss']:.6f}\n")
        f.write(f"Best w1: {best_weights['w1']:.6f}\n")
        f.write(f"Best w2: {best_weights['w2']:.6f}\n")
        f.write(f"Best w3: {best_weights['w3']:.6f}\n")   


    return savePath   

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
