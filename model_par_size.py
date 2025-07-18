import torch
from model_Louck_light import NetworkBasic  # 替换成你的文件路径
import slayerSNN as snn

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_model_size(model):
    """估算模型内存占用（单位：MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size / (1024 ** 2)  # bytes -> MB

if __name__ == "__main__":
    # 模拟参数文件（或加载真实 network.yaml）
    netParams = snn.params('./nMnist/network.yaml')

    # 创建模型
    model = NetworkBasic(netParams)
    # 🔍 打印模型结构
    print("\n📦 模型结构如下:\n")
    print(model)


    # 输出参数信息
    total_params, trainable_params = count_parameters(model)
    model_size_mb = estimate_model_size(model)

    print(f"✅ Total Parameters: {total_params:,}")
    print(f"✅ Trainable Parameters: {trainable_params:,}")
    print(f"💾 Estimated Model Size: {model_size_mb:.4f} MB ({model_size_mb * 1024:.2f} KB)")

