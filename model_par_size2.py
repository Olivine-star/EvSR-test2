import torch
from model_Louck_light import NetworkBasic
import slayerSNN as snn


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total} ({total / 1e6:.3f} M)")

if __name__ == '__main__':
    # 加载 SlayerSNN 的网络配置参数
    netParams = snn.params('./nMnist/network.yaml')

    # 创建模型
    model = NetworkBasic(netParams)

    # 多GPU时模型会被包装在 DataParallel 内部，要取出 .module 来访问真实模型结构
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    # 打印参数数量
    count_parameters(model)
