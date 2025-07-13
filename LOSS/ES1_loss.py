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
    loss_total = loss + 5 * loss_ecm
    #loss_total = loss + loss_ecm
    return loss_total, loss, loss_ecm



def validation_loss(output, target, shape):
    MSE = nn.MSELoss()
    loss = MSE(output, target)
    loss_ecm = sum([
        MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
            torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
        for i in range(shape[2] // 50)
    ])
    loss_total = loss + loss_ecm
    return loss_total, loss, loss_ecm