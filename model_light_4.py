import torch
import slayerSNN as snn
from utils.utils import getNeuronConfig
import numpy as np


class NetworkBasic(torch.nn.Module):
    """
    这个模型就由trainNmnist传入两个参数：netParams和传入spikeInput
    netParams = snn.params('network.yaml'),调用模型类，创建网络对象,m = NetworkBasic(netParams)
    output = m(eventLr)，eventLr就是传入forward的参数spikeInput
    """
    #
    def __init__(self, netParams,
                 theta=[30, 100],
                 tauSr=[1, 4],
                 tauRef=[1, 4],
                 scaleRef=[1, 1],
                 tauRho=[1, 10],
                 scaleRho=[10, 100]):
        super(NetworkBasic, self).__init__()

        """
        每一层神经元的行为由 getNeuronConfig(...) 函数配置，如阈值、衰减时间等，是 SNN 特有的。
        这是为了给每一层设置不同的神经元行为参数，因为在 SNN 中：神经元不再是普通的 ReLU 激活单元，而是具有生物启发特性的脉冲神经元；

        参数如 theta（发放阈值）、tauSr（突触响应时间常数）、tauRef（不应期时间）、scaleRho（电压或重置幅值）等，会影响神经元的发放频率、时间响应、抑制机制等。

        ✅ 目的：通过为不同层设置不同神经元行为，让网络在不同阶段能学习到更合适的时间动态。
        修改成两层
        
        """
        # 依次将三个不同层的神经元参数添加到空列表 self.neuron_config 中
        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        
        # 取参数列表的第0个元素，即第一个神经元参数，这个配置会被传入 snn.layer(...)，这些参数会对应赋值给其内部的神经元模型
        # neuronDesc (slayerParams.yamlParams): spiking neuron descriptor.
        # simulationDesc (slayerParams.yamlParams): simulation descriptor
        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        

        # 是卷积层，由配置了对应参数的给自的 snn.layer 提供（slayer.py中定义了conv函数，就是调用slayer.py中conv函数，想要什么层，就在slayer.py中定义），带有脉冲特性。
        self.conv1 = self.slayer1.conv(2, 8, 5, padding=2)
        
        self.upconv1 = self.slayer2.convTranspose(8, 2, kernelSize=4, stride=4)

    # 这段 forward 函数是 NetworkBasic 的前向传播逻辑，用于对输入的 脉冲张量（事件数据） 进行 时空建模和上采样重建。
    def forward(self, spikeInput):
        # 通过 slayer1.psp() 对输入进行电压膜电位建模
        # print("=================================================================")
        # print(spikeInput.shape)
        psp1 = self.slayer1.psp(spikeInput)

        # 输入为 [B, C, H, W, T] 形状的 5D 张量，表示一批事件数据（脉冲流）。
        # H 和 W 仍然是空间位置，代表传感器像素网格上的坐标。
        # 获取输入的维度
        B, C, H, W, T = spikeInput.shape
        # 把时间维度移到前面
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        # 合并时间和通道，变成一堆图
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        # 空间上采样 ×2. interpolate 是 PyTorch 的插值函数，这里用 bilinear 模式对 空间维度 H 和 W 上采样 2 倍
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=4, mode='bilinear')
        # 再转回原来的维度顺序.将前面合并的 C*T 再拆开为 C 和 T，变成 [B, C, T, 2H, 2W]，
        # 然后 permute 成 [B, C, 2H, 2W, T]，与 SNN 的标准输入一致。
        psp1_1 = psp1_1.reshape(B, C, T, 4*H, 4*W).permute((0, 1, 3, 4, 2))

        # 将 PSP 结果输入脉冲卷积层，并用阈值函数转换为脉冲输出（脉冲表示事件是否激活）。
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        # 对上一层的脉冲输出继续进行 PSP，再卷积、再脉冲。
        
        # PSP 后上采样，然后与前面旁路上采样的 psp1_1 相加(像 ResNet 的残差，加入一条细节旁路路径)，再经过脉冲发放，输出最终脉冲结果。
        spikes_layer_2 = self.slayer2.spike(self.upconv1(self.slayer2.psp(spikes_layer_1)) + psp1_1)

        # 这是模型对输入低分辨率事件张量的超分重建输出。
        # 最终输出大小：从 32x32 → 34x34（只插值，不改变通道和时间维度）
        # 🚨 修正插值维度错误
        B, C, H, W, T = spikes_layer_2.shape

        # [B, C, H, W, T] → [B, C, T, H, W]
        spikes_layer_2 = spikes_layer_2.permute(0, 1, 4, 2, 3)
        # [B, C, T, H, W] → [B*T, C, H, W]
        spikes_layer_2 = spikes_layer_2.reshape(B * T, C, H, W)

        # ✅ 空间插值到 34×34
        spikes_layer_2 = torch.nn.functional.interpolate(spikes_layer_2, size=(34, 34), mode='bilinear', align_corners=False)

        # [B*T, C, 34, 34] → [B, T, C, 34, 34]
        spikes_layer_2 = spikes_layer_2.view(B, T, C, 34, 34)
        # → [B, C, 34, 34, T]
        spikes_layer_2 = spikes_layer_2.permute(0, 2, 3, 4, 1)

        return spikes_layer_2




if __name__ == '__main__':
    import os
    from slayerSNN.spikeFileIO import event

    def readNpSpikes(filename, timeUnit=1e-3):
        npEvent = np.load(filename)
        return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x = readNpSpikes(r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\1.npy")
    # 转化事件数据为脉冲张量，极性提到前面当做两个通道（普通图像转张量的维度顺序都是第一维度为通道维度）
    x = x.toSpikeTensor(torch.zeros((2, 17, 17, 350)))
    print(x)
    # 因为一个脉冲的维度是5个维度，所以在第 0 个维度上插入一个大小为 1 的新维度。
    x = torch.unsqueeze(x, dim=0).cuda()
    print(x)

    netParams = snn.params('./nMnist/network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).cuda()
    with torch.no_grad():
        out = m(x)
    print((out == 0).sum(), (out == 1).sum(), ((out != 0) & (out != 1)).sum())

    # 如果是后者，这可能是正常的，是指输出了幅值为2 3 4的脉冲，而不都是单位脉冲。我记得slayersnn库输出的spike好像是有幅值的。
    # 同样，我们输入的spike有些也有幅值，由于我们将原始事件流沿时间维度堆叠到tSample个channel（e.g., tSample=350 for nMNIST dataset），
    # 在压缩过程中，如短时间内同一像素点触发多个event，会堆叠成一个倍数幅值的spike。