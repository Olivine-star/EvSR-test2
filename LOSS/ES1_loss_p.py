import torch
import torch.nn as nn




def training_loss(output, target, shape):
    MSE = nn.MSELoss()

    # 基础MSE损失
    loss = MSE(output, target)

    # 时间块的ECM损失
    loss_ecm = sum([
        MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
            torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
        for i in range(shape[2] // 50)
    ])
    # 加入极性差异的损失（正负通道）
    loss_polarity = MSE(output[:, 0, ...], target[:, 0, ...]) + MSE(output[:, 1, ...], target[:, 1, ...])

    # loss_total = loss + 5 * loss_ecm + 0.1 * loss_polarity
    # loss_total = loss  + 0.1 * loss_polarity
    loss_total = loss + 5 * loss_ecm +  loss_polarity

    #loss_total = loss + loss_ecm
    return loss_total, loss, loss_ecm, loss_polarity



def validation_loss(output, target, shape):
    MSE = nn.MSELoss()
    loss = MSE(output, target)
    loss_ecm = sum([
        MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
            torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
        for i in range(shape[2] // 50)
    ])
    # 加入极性差异的损失（正负通道）
    loss_polarity = MSE(output[:, 0, ...], target[:, 0, ...]) + MSE(output[:, 1, ...], target[:, 1, ...])

    # loss_total = loss + loss_ecm + 0.1 * loss_polarity
    # loss_total = loss + 0.1 * loss_polarity
    loss_total = loss + loss_ecm  +  loss_polarity

    return loss_total, loss, loss_ecm, loss_polarity