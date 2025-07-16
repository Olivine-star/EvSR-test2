import torch
import slayerSNN as snn
from utils.utils import getNeuronConfig
import numpy as np

# 就改了网络层顺序和参数
class NetworkBasic(torch.nn.Module):
    
    
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        
        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        self.neuron_config.append(getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2], scaleRho=scaleRho[2]))

        
        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        
        self.conv1 = self.slayer1.conv(2, 8, 5, padding=2)
        self.upconv1 = self.slayer2.convTranspose(8, 2, kernelSize=2, stride=2)
        self.conv2 = self.slayer3.conv(2, 2, 3, padding=1)
        

    
    def forward(self, spikeInput):
        
        psp1 = self.slayer1.psp(spikeInput)

        
        B, C, H, W, T = spikeInput.shape
        
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        
        psp1_1 = psp1_1.reshape(B, C, T, 2*H, 2*W).permute((0, 1, 3, 4, 2))

        
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
       
        spikes_layer_2 = self.slayer2.spike(self.upconv1(self.slayer2.psp(spikes_layer_1)))
        
        spikes_layer_3 = self.slayer3.spike(self.conv2(self.slayer3.psp(spikes_layer_2)) + psp1_1)

        
        return spikes_layer_3


if __name__ == '__main__':
    import os
    from slayerSNN.spikeFileIO import event

    def readNpSpikes(filename, timeUnit=1e-3):
        npEvent = np.load(filename)
        return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x = readNpSpikes(r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\1.npy")
    
    x = x.toSpikeTensor(torch.zeros((2, 17, 17, 350)))
    print(x)
    
    x = torch.unsqueeze(x, dim=0).cuda()
    print(x)

    netParams = snn.params('./nMnist/network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).cuda()
    with torch.no_grad():
        out = m(x)
    print((out == 0).sum(), (out == 1).sum(), ((out != 0) & (out != 1)).sum())


# ===================基于SNN,只修改模型结构===================================



# import torch
# import torch.nn.functional as F
# import slayerSNN as snn
# from utils.utils import getNeuronConfig
# import numpy as np


# class NetworkBasic(torch.nn.Module):
#     def __init__(self, netParams,
#                  theta=[30, 50, 100],
#                  tauSr=[1, 2, 4],
#                  tauRef=[1, 2, 4],
#                  scaleRef=[1, 1, 1],
#                  tauRho=[1, 1, 10],
#                  scaleRho=[10, 10, 100]):
#         super(NetworkBasic, self).__init__()

#         # 创建神经元配置
#         self.neuron_config = []
#         for i in range(3):
#             self.neuron_config.append(getNeuronConfig(
#                 theta=theta[i], tauSr=tauSr[i],
#                 tauRef=tauRef[i], scaleRef=scaleRef[i],
#                 tauRho=tauRho[i], scaleRho=scaleRho[i]))

#         # 三层 SNN 模块
#         self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])  # SDNN-C1
#         self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])  # SDNN-CT
#         self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])  # SDNN-C2

#         # 模块 SDNN-C1: Conv2d, 输入2通道，输出8通道，kernel=(5,5,1)
#         self.conv1 = self.slayer1.conv(2, 8, kernelSize=(5, 5), stride=(1, 1), padding=(2, 2))

#         # 模块 SDNN-CT: ConvTranspose2d, 输入8通道，输出2通道
#         self.deconv = self.slayer2.convTranspose(8, 2, kernelSize=(2, 2), stride=(2, 2), padding=(2, 2))

#         # 模块 SDNN-C2: Conv2d, 输入2通道，输出2通道，kernel=(1,1,1)
#         self.conv2 = self.slayer3.conv(2, 2, kernelSize=(1, 1), stride=1)

#         # Dropout（时间一致性）
#         self.drop2 = self.slayer2.dropout(p=0.1)
#         self.drop3 = self.slayer3.dropout(p=0.1)

#     def forward(self, spikeInput):
#         # SDNN-C1
#         spike1 = self.slayer1.spike(self.conv1(self.slayer1.psp(spikeInput)))

#         # SDNN-CT（包含上采样 + dropout）
#         psp2 = self.slayer2.psp(spike1)
#         upsampled = F.interpolate(psp2, scale_factor=(2, 2, 1), mode='trilinear', align_corners=False)
#         deconv_out = self.deconv(upsampled)
#         deconv_out = self.drop2(deconv_out)
#         spike2 = self.slayer2.spike(deconv_out)

#         # SDNN-C2（包含 dropout）
#         psp3 = self.slayer3.psp(spike2)
#         conv_out = self.conv2(psp3)
#         conv_out = self.drop3(conv_out)
#         spike3 = self.slayer3.spike(conv_out)

#         return spike3


# # 测试主程序
# if __name__ == '__main__':
#     import os
#     from slayerSNN.spikeFileIO import event

#     def readNpSpikes(filename, timeUnit=1e-3):
#         npEvent = np.load(filename)
#         return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#     # 读取 spike tensor（这里根据你的路径）
#     x = readNpSpikes(r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\1.npy")
#     x = x.toSpikeTensor(torch.zeros((2, 17, 17, 350)))  # (C, H, W, T)
#     x = torch.unsqueeze(x, dim=0).cuda()  # 添加 batch 维度

#     # 加载参数和模型
#     netParams = snn.params('./nMnist/network.yaml')
#     model = NetworkBasic(netParams)
#     model = torch.nn.DataParallel(model).cuda()

#     # 前向推理
#     with torch.no_grad():
#         out = model(x)

#     # 打印输出统计信息
#     print("==> 输出统计：")
#     print("0 值数量：", (out == 0).sum().item())
#     print("1 值数量（脉冲）：", (out == 1).sum().item())
#     print("其他值（异常）：", ((out != 0) & (out != 1)).sum().item())


































# ======================================================


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SDNN_Network(nn.Module):
#     def __init__(self, time_window=350):
#         super(SDNN_Network, self).__init__()

#         # Time window T
#         self.time_window = time_window

#         # SDNN-C1: Conv3D with 2→8, kernel=(5,5,1), stride=(1,1,1), padding=(2,2,0)
#         self.conv1 = nn.Conv3d(in_channels=2, out_channels=8,
#                                kernel_size=(5, 5, 1),
#                                stride=(1, 1, 1),
#                                padding=(2, 2, 0))

#         # SDNN-CT: Transposed Conv3D with 8→2, kernel=(2,2,1), stride=(2,2,1)
#         self.deconv = nn.ConvTranspose3d(in_channels=8, out_channels=2,
#                                          kernel_size=(2, 2, 1),
#                                          stride=(2, 2, 1))

#         # Dropout after deconv
#         self.dropout = nn.Dropout3d(0.1)

#         # SDNN-C2: Conv3D with 2→2, kernel=(1,1,1), stride=1
#         self.conv2 = nn.Conv3d(in_channels=2, out_channels=2,
#                                kernel_size=(1, 1, 1),
#                                stride=(1, 1, 1))

#     def delta_encode(self, x):
#         """
#         Compute temporal difference: Δx[t] = x[t] - x[t-1]
#         Input: x [B, C, H, W, T]
#         Output: same shape [B, C, H, W, T]
#         """
#         delta = x[..., 1:] - x[..., :-1]
#         zero_pad = torch.zeros_like(x[..., :1])
#         return torch.cat((zero_pad, delta), dim=-1)

#     def forward(self, x):
#         """
#         Input: x [B, 2, H, W, T] (2 = polarity channels)
#         Output: [B, 2, H_out, W_out, T]
#         """
#         # Sigma-Delta: ΔT encoding
#         x = self.delta_encode(x)

#         # Change to 5D [B, C, T, H, W] for Conv3D (PyTorch expects [B, C, D, H, W])
#         x = x.permute(0, 1, 4, 2, 3)

#         # SDNN-C1
#         x = F.relu(self.conv1(x))  # shape: [B, 8, T, H, W]

#         # SDNN-CT
#         x = self.deconv(x)         # upsampled: [B, 2, T, 2H, 2W]
#         x = self.dropout(x)
#         x = F.relu(x)

#         # SDNN-C2
#         x = F.relu(self.conv2(x))  # shape: [B, 2, T, 2H, 2W]

#         # Restore shape to [B, 2, H, W, T]
#         x = x.permute(0, 1, 3, 4, 2)
#         return x




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SDNN_LIFNeuron(nn.Module):
#     def __init__(self, alpha=0.95, threshold=1.0):
#         super().__init__()
#         self.alpha = alpha
#         self.threshold = threshold

#     def forward(self, x):
#         """
#         x: Tensor [B, C, T, H, W]
#         Implements:
#         - Leaky Integration: y[t] = (1-α)y[t-1] + x[t]
#         - Spiking: s[t] = 1 if y[t] ≥ θ
#         - Reset: y[t] = y[t] * (1 - s[t])
#         """
#         B, C, T, H, W = x.shape
#         y = torch.zeros_like(x).to(x.device)
#         s = torch.zeros_like(x).to(x.device)

#         for t in range(T):
#             if t == 0:
#                 y[:, :, t] = x[:, :, t]
#             else:
#                 y[:, :, t] = (1 - self.alpha) * y[:, :, t - 1] + x[:, :, t]

#             s[:, :, t] = (y[:, :, t] >= self.threshold).float()
#             y[:, :, t] *= (1 - s[:, :, t])  # spike-dependent reset

#         return s


# class SDNN_Network(nn.Module):
#     def __init__(self, time_window=350):
#         super(SDNN_Network, self).__init__()
#         self.time_window = time_window

#         # Neuron layer for memory and spiking
#         self.lif_neuron1 = SDNN_LIFNeuron(alpha=0.95, threshold=1.0)
#         self.lif_neuron2 = SDNN_LIFNeuron(alpha=0.95, threshold=1.0)

#         # SDNN-C1: Conv3D 2→8
#         self.conv1 = nn.Conv3d(2, 8, kernel_size=(5, 5, 1), stride=(1, 1, 1), padding=(2, 2, 0))

#         # SDNN-CT: Deconv3D 8→2
#         self.deconv = nn.ConvTranspose3d(8, 2, kernel_size=(2, 2, 1), stride=(2, 2, 1))

#         # Dropout
#         self.dropout = nn.Dropout3d(0.1)

#         # SDNN-C2: Conv3D 2→2 (1x1 kernel)
#         self.conv2 = nn.Conv3d(2, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))

#     def delta_encode(self, x):
#         """
#         ΔT Temporal Difference Encoding
#         x: [B, C, T, H, W] -> return same shape
#         """
#         delta = x[:, :, 1:] - x[:, :, :-1]
#         zero_pad = torch.zeros_like(x[:, :, :1])
#         return torch.cat((zero_pad, delta), dim=2)  # dim=2 is T

#     def forward(self, x):
#         """
#         x: [B, 2, H, W, T]
#         output: [B, 2, H_up, W_up, T]
#         """
#         # Permute to [B, C, T, H, W] for Conv3D
#         x = x.permute(0, 1, 4, 2, 3)

#         # === SDNN-C1 ===
#         x = F.relu(self.conv1(x))
#         x = self.lif_neuron1(x)

#         # === ΔT after first neuron ===
#         x = self.delta_encode(x)

#         # === SDNN-CT (deconv + dropout) ===
#         x = self.deconv(x)
#         x = self.dropout(x)
#         x = F.relu(x)

#         # === SDNN-C2 ===
#         x = self.conv2(x)
#         x = self.lif_neuron2(x)

#         # Permute back to [B, 2, H, W, T]
#         x = x.permute(0, 1, 3, 4, 2)
#         return x


# if __name__ == '__main__':
#     model = SDNN_Network(time_window=350).cuda()
#     x = torch.randn(1, 2, 17, 17, 350).cuda()  # [B, 2, H, W, T]
    
#     with torch.no_grad():
#         out = model(x)
    
#     print("Output shape:", out.shape)  # Expected: [1, 2, 34, 34, 350]
