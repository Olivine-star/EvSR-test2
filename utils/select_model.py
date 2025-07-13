import os
import torch
from utils.ckpt import checkpoint_restore
from utils.utils import getEventFromTensor
import numpy as np

def run_multiple_models(model, testLoader, base_ckpt_path, base_save_path, device='cuda'):
    """
    循环加载前10个模型权重，对测试数据进行推理并分别保存结果。
    参数:
        model: 已构建好的模型（未加载权重）
        testLoader: PyTorch 的 DataLoader，提供测试数据
        base_ckpt_path: 模型权重所在的目录，如 ".../bs64_lr0.1_ep30_cuda0_20250713_170414"
        base_save_path: 保存推理结果的根目录，如 ".../HRPre"
        device: 使用设备，默认是 'cuda'
    """
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    for i in range(10):  # 只用前10个模型
        ckpt_file = os.path.join(base_ckpt_path, f'ckpt{i}.pth')
        if not os.path.exists(ckpt_file):
            print(f"[!] 模型文件不存在: {ckpt_file}")
            continue

        print(f"[✓] 加载模型: {ckpt_file}")
        model, epoch0 = checkpoint_restore(model, ckpt_file, name='', device=device)

        # 每个模型的保存路径
        model_save_path = os.path.join(base_save_path, f'model_{i}')
        os.makedirs(model_save_path, exist_ok=True)

        for k, (eventLr, _, path) in enumerate(testLoader):
            with torch.no_grad():
                eventLr = eventLr.to(device)

                # 分通道预测
                eventLr_pos = eventLr[:, 0:1, ...]
                eventLr_neg = eventLr[:, 1:2, ...]
                output_pos = model(eventLr_pos)
                output_neg = model(eventLr_neg)
                output = torch.cat([output_pos, output_neg], dim=1)

                # 转为事件数据
                eventList = getEventFromTensor(output)
                e = eventList[0]
                e = e[:, [0, 2, 1, 3]]  # 转为[t, x, y, p]

                # 保存路径：savepath/model_i/class/filename.npy
                output_file_path = os.path.join(model_save_path, path[0])
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                np.save(output_file_path, e.astype(np.int32))

                if k % 100 == 0:
                    print(f"模型{i} 推理进度：{k}/{len(testLoader)}")
