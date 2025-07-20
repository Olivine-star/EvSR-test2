import torch
import torch.nn as nn

class LearnableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化可学习的 log(sigma^2)，确保正数
        self.log_w1 = nn.Parameter(torch.tensor(0.0))  # for MSE loss
        self.log_w2 = nn.Parameter(torch.tensor(0.0))  # for ECM loss
        self.log_w3 = nn.Parameter(torch.tensor(0.0))  # for polarity loss
        self.MSE = nn.MSELoss()

    def forward(self, output, target, shape):
        loss = self.MSE(output, target)

        loss_ecm = sum([
            self.MSE(torch.sum(output[:, :, :, :, i * 50:(i + 1) * 50], dim=4),
                     torch.sum(target[:, :, :, :, i * 50:(i + 1) * 50], dim=4))
            for i in range(shape[2] // 50)
        ])

        loss_polarity = self.MSE(output[:, 0, ...], target[:, 0, ...]) + \
                        self.MSE(output[:, 1, ...], target[:, 1, ...])

        # 使用 softplus 确保正值（也可以用 torch.exp）
        w1 = torch.exp(-self.log_w1)
        w2 = torch.exp(-self.log_w2)
        w3 = torch.exp(-self.log_w3)

        loss_total = w1 * loss + w2 * loss_ecm + w3 * loss_polarity + \
                     (self.log_w1 + self.log_w2 + self.log_w3)  # Regularization term to prevent zero weights

        return loss_total, loss, loss_ecm, loss_polarity
