# import os
# import numpy as np
# import torch
# from tqdm import tqdm

# # 输入输出路径
# input_root = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\NFS-50ms\HR"
# output_root = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\NFS-50ms\HR-guiyi"
# os.makedirs(output_root, exist_ok=True)

# target_bins = 50  # 时间戳范围 [0, 49]

# # 遍历文件夹
# for folder_name in tqdm(os.listdir(input_root), desc="Processing Folders"):
#     input_folder = os.path.join(input_root, folder_name)
#     output_folder = os.path.join(output_root, folder_name)
#     if not os.path.isdir(input_folder):
#         continue
#     os.makedirs(output_folder, exist_ok=True)

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".npy"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)

#             # 读取事件数据
#             events = np.load(input_path)
#             if events.shape[1] != 4:
#                 print(f"[跳过] 文件格式错误: {input_path}")
#                 continue

#             t = torch.tensor(events[:, 0], dtype=torch.float32, device='cuda')
#             t_min = t.min()
#             t_max = t.max()
#             if t_max == t_min:
#                 t_norm = torch.zeros_like(t, dtype=torch.int32)
#             else:
#                 t_norm = ((t - t_min) / (t_max - t_min) * (target_bins - 1)).to(torch.int32)

#             x = torch.tensor(events[:, 1], dtype=torch.int32, device='cuda')
#             y = torch.tensor(events[:, 2], dtype=torch.int32, device='cuda')
#             p = torch.tensor(events[:, 3], dtype=torch.int32, device='cuda')

#             result = torch.stack([t_norm, x, y, p], dim=1).cpu().numpy().astype(np.int32)

#             # 保存
#             np.save(output_path, result)




import os
import numpy as np
import torch
from tqdm import tqdm

# 参数配置
input_root = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\NFS-50ms\HR"
output_root = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\NFS-50ms\HR-guiyi"
target_bins = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 遍历所有子目录
for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    for fname in tqdm(files, desc=f"Processing folder {folder}"):
        file_idx = int(os.path.splitext(fname)[0])  # 文件编号
        offset = (file_idx - 1) * target_bins

        fpath = os.path.join(folder_path, fname)
        events_np = np.load(fpath).astype(np.int32)
        events = torch.tensor(events_np, dtype=torch.float32, device=device)

        t = events[:, 0]
        t_min = t.min()
        t_max = t.max()

        # 避免除以 0 的情况（恒定时间戳）
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (target_bins - 1)).to(torch.int32) + offset
        else:
            t_norm = torch.full_like(t, offset, dtype=torch.int32)

        events = events.to(torch.int32)
        events[:, 0] = t_norm.cpu()

        save_path = os.path.join(output_folder, fname)
        np.save(save_path, events.cpu().numpy())
