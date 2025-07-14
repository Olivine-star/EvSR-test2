import torch
import torch.nn as nn
import slayerSNN as snn
from utils.utils import getNeuronConfig


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(1, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W, T]
        attn_map = torch.sigmoid(self.conv(avg_out))
        return x * attn_map


class NetworkBasic(nn.Module):
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        self.neuron_config = [
            getNeuronConfig(theta=theta[i], tauSr=tauSr[i], tauRef=tauRef[i],
                            scaleRef=scaleRef[i], tauRho=tauRho[i], scaleRho=scaleRho[i])
            for i in range(3)
        ]

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        self.conv1 = self.slayer1.conv(1, 8, 5, padding=2)
        self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)
        self.upconv1 = self.slayer3.convTranspose(8, 1, kernelSize=2, stride=2)

        self.attn1 = SpatialAttention3D()
        self.attn2 = SpatialAttention3D()

    def forward(self, spikeInput):
        psp1 = self.attn1(self.slayer1.psp(spikeInput))
        B, C, H, W, T = spikeInput.shape
        psp1_1 = psp1.permute(0, 1, 4, 2, 3).reshape(B, C * T, H, W)
        psp1_1 = nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        psp1_1 = psp1_1.reshape(B, C, T, 2 * H, 2 * W).permute(0, 1, 3, 4, 2)

        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        spikes_layer_2 = self.attn2(self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1))))
        spikes_layer_3 = self.slayer3.spike(self.upconv1(self.slayer3.psp(spikes_layer_2)) + psp1_1)

        return spikes_layer_3


class NetworkBasic1(nn.Module):
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic1, self).__init__()

        self.neuron_config = [
            getNeuronConfig(theta=theta[i], tauSr=tauSr[i], tauRef=tauRef[i],
                            scaleRef=scaleRef[i], tauRho=tauRho[i], scaleRho=scaleRho[i])
            for i in range(3)
        ]

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        self.conv1 = self.slayer1.conv(2, 8, 5, padding=2)
        self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)
        self.upconv1 = self.slayer3.convTranspose(8, 2, kernelSize=2, stride=2)

        self.attn1 = SpatialAttention3D()
        self.attn2 = SpatialAttention3D()

    def forward(self, spikeInput):
        psp1 = self.attn1(self.slayer1.psp(spikeInput))
        B, C, H, W, T = spikeInput.shape
        psp1_1 = psp1.permute(0, 1, 4, 2, 3).reshape(B, C * T, H, W)
        psp1_1 = nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        psp1_1 = psp1_1.reshape(B, C, T, 2 * H, 2 * W).permute(0, 1, 3, 4, 2)

        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        spikes_layer_2 = self.attn2(self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1))))
        spikes_layer_3 = self.slayer3.spike(self.upconv1(self.slayer3.psp(spikes_layer_2)) + psp1_1)

        return spikes_layer_3


class DualBranchWithGuidance(nn.Module):
    def __init__(self, netParams):
        super(DualBranchWithGuidance, self).__init__()
        self.pos_branch = NetworkBasic(netParams)
        self.neg_branch = NetworkBasic(netParams)
        self.guidance_branch = NetworkBasic1(netParams)

    def forward(self, spikeInput):
        spike_pos = spikeInput[:, 0:1, ...]  # [B, 1, H, W, T]
        spike_neg = spikeInput[:, 1:2, ...]  # [B, 1, H, W, T]
        #spike_merge = torch.sum(spikeInput, dim=1, keepdim=True)  # [B, 1, H, W, T]

        out_pos = self.pos_branch(spike_pos)
        out_neg = self.neg_branch(spike_neg)
        out_fused = torch.cat([out_pos, out_neg], dim=1)  # [B, 2, H, W, T]

        out_guide = self.guidance_branch(spikeInput)  # [B, 1, H, W, T]

        # 交集融合（指导信号）
        #soft_mask = torch.sigmoid(out_guide)  # 连续值 ∈ (0, 1)
        #out_fused = out_fused * soft_mask.expand(-1, 2, -1, -1, -1)
        soft_mask = torch.sigmoid(out_guide)  # [B, 1, H, W, T]
        soft_mask = soft_mask.expand(-1, 2, -1, -1, -1)

        # 取最大或加权融合（推荐）
        out_fused = out_fused + soft_mask

        print(f"[Debug] out_fused shape: {out_fused.shape}, nonzero: {(out_fused > 0).sum().item()}")
        #print(f"[Debug] out_guide shape: {out_guide.shape}, nonzero: {(out_guide > 0).sum().item()}")

        return out_fused
