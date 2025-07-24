# import torch
# from ptflops import get_model_complexity_info
# from model import NetworkBasic
# import slayerSNN as snn


# def count_flops():
#     # 加载 SNN 配置
#     netParams = snn.params('./nMnist/network.yaml')

#     # 创建模型并设置为 evaluation 模式
#     model = NetworkBasic(netParams)
#     model.eval()

#     # input_shape: 这里的尺寸是你模型 forward 的输入尺寸，不包括 batch size
#     input_shape = (2, 17, 17, 350)  # 对应 [C, H, W, T]
    
#     # ptflops 只支持 NCHW，变换成 [C, T, H, W] => reshape 为 [C*T, H, W]
#     c, h, w, t = input_shape
#     flops_shape = (c * t, h, w)  # 注意不要包含 batch dim

#     # 包装一下模型 forward，让它接受 4D 输入
#     class Wrapper(torch.nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, x):
#             # 将 4D x (C*T, H, W) 还原为原始 5D spike tensor: [B, C, H, W, T]
#             B, CT, H, W = x.shape
#             x = x.view(B, c, t, H, W).permute(0, 1, 3, 4, 2)  # [B, C, H, W, T]
#             return self.model(x)

#     wrapped_model = Wrapper(model)

#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(wrapped_model, (c*t, h, w), as_strings=True, print_per_layer_stat=True)
#         print(f"FLOPs: {macs}")
#         print(f"Params: {params}")


# if __name__ == '__main__':
#     count_flops()



"""
本脚本只估算了两层卷积，不包含其他如 PSP/spike 处理，因为它们是事件神经网络特有操作，不适用标准 FLOPs 模型。
spike/spikePSP 等神经动力学过程不计入 FLOPs（通常也无法准确建模）
"""



# def estimate_flops_conv3d(Cin, Cout, K, H, W, T):
#     """
#     估算 3D 卷积 FLOPs：
#     每个输出位置：Cout × (Cin × K × K) 次乘法 + 相同数量加法 ≈ 2 × Cout × Cin × K × K
#     总 FLOPs = 2 × Cout × Cin × K² × H × W × T
#     """
#     return 2 * Cout * Cin * (K ** 2) * H * W * T


# def estimate_params_conv(Cin, Cout, K):
#     """估算普通 2D 卷积层的参数数量"""
#     return Cout * Cin * K * K


# if __name__ == "__main__":
#     # 输入张量尺寸 [B, C=2, H=17, W=17, T=350]
#     Cin1, Cout1, K1 = 2, 8, 5  # conv1
#     Cin2, Cout2, K2 = 8, 1, 2  # convTranspose

#     # 层1：原始分辨率
#     H1, W1, T = 17, 17, 350

#     # 层2：经过上采样后的分辨率
#     H2, W2 = H1 * 2, W1 * 2

#     # 计算 FLOPs
#     flops_conv1 = estimate_flops_conv3d(Cin1, Cout1, K1, H1, W1, T)
#     flops_upconv1 = estimate_flops_conv3d(Cin2, Cout2, K2, H2, W2, T)
#     total_flops = flops_conv1 + flops_upconv1

#     # 计算参数量
#     params_conv1 = estimate_params_conv(Cin1, Cout1, K1)
#     params_upconv1 = estimate_params_conv(Cin2, Cout2, K2)
#     total_params = params_conv1 + params_upconv1

#     # 输出结果
#     print(f"[conv1] FLOPs:        {flops_conv1 / 1e6:.2f} MFLOPs, Params: {params_conv1}")
#     print(f"[upconv1] FLOPs:      {flops_upconv1 / 1e6:.2f} MFLOPs, Params: {params_upconv1}")
#     print(f"Total FLOPs:          {total_flops / 1e6:.2f} MFLOPs")
#     print(f"Total Trainable Params: {total_params} ({total_params / 1e6:.3f} M)")



"""
这是手动计算的，自动的识别不出Slayer 的卷积层
"""

import torch
from model import NetworkBasic
import slayerSNN as snn

# 🔧 填写实际输入尺寸
C_in, H_in, W_in, T = 2, 34, 34, 350  # 双极性通道，180x240，1500时间步
input_shape = (C_in, H_in, W_in, T)

# ✅ 加载 Slayer 参数
netParams = snn.params('./nMnist/network.yaml')
model = NetworkBasic(netParams).cuda()

T = 350

def count_conv_flops(H, W, Cin, Cout, K, stride=1, upsample=False):
    if upsample:
        H, W = H * stride, W * stride
    flops = 2 * H * W * Cin * K * K * Cout * T  # ✅ 注意乘 T
    return flops, H, W

# 👉 逐层统计（以下仅举例，需按你模型实际结构填写）
total_flops = 0
total_params = 0

# conv1: in=2, out=8, k=5, stride=1, no upsampling
flops, H1, W1 = count_conv_flops(H_in, W_in, 2, 8, K=5)
params = 2 * 8 * 5 * 5  # Cin * Cout * Kh * Kw
print(f"[conv1] FLOPs: {flops/1e9:.2f} GFLOPs, Params: {params}")
total_flops += flops
total_params += params

# conv2: in=8, out=8, k=3, stride=1
flops, H2, W2 = count_conv_flops(H1, W1, 8, 8, K=3)
params = 8 * 8 * 3 * 3
print(f"[conv2] FLOPs: {flops/1e9:.2f} GFLOPs, Params: {params}")
total_flops += flops
total_params += params

# upconv1: TransposedConv, in=8, out=2, k=2, stride=2
flops, H3, W3 = count_conv_flops(H2, W2, 8, 2, K=2, stride=2, upsample=True)
params = 8 * 2 * 2 * 2
print(f"[upconv1] FLOPs: {flops/1e9:.2f} GFLOPs, Params: {params}")
total_flops += flops
total_params += params

# ✅ 输出总计
print(f"✅ Total FLOPs: {total_flops/1e9:.2f} GFLOPs")
print(f"✅ Total Trainable Params: {total_params} ({total_params/1e6:.3f} M)")




