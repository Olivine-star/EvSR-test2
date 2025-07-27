# import numpy as np
# import os
# import torch
# import torch.nn.functional as F

# # 配置参数
# window = 1024
# sliding_window = 512
# H, W = 260, 346  # 分辨率（从 txt 文件第一行读取）
# mse_save_path = r"C:\Users\chxu4146\Project\EvSR-result\NFS\7-26\baseline-50ms(guiyi)\HRPre\2\mse.txt"

# gt_path = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\dataset\SR_Test\HR\2\IR.txt"
# pred_path = r"C:\Users\chxu4146\Project\EvSR-result\NFS\7-26\baseline-50ms(guiyi)\HRPre\2\IR.txt"

# def load_events(txt_path):
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()

#     W_loaded, H_loaded = map(int, lines[0].strip().split())
#     assert W_loaded == W and H_loaded == H, "尺寸不一致！"

#     events = []
#     for line in lines[1:]:
#         line = line.strip()
#         if not line:  # 跳过空行
#             continue
#         parts = line.split()
#         if len(parts) != 4:
#             print(f"[警告] 无效行被跳过: {line}")
#             continue
#         try:
#             t, x, y, p = map(int, parts)
#             events.append([t, x, y, p])
#         except ValueError:
#             print(f"[警告] 无法转换为整数的行被跳过: {line}")
#             continue

#     return np.array(events)


# def events_to_count_map(events, H, W):
#     """
#     输入：[N, 4] (t, x, y, p)
#     输出：事件计数图 [2, H, W]
#     """
#     count_map = np.zeros((2, H, W), dtype=np.float32)
#     for t, x, y, p in events:
#         if 0 <= x < W and 0 <= y < H:
#             count_map[p, y, x] += 1
#     return count_map

# def compute_mse_for_all_windows(events1, events2, window, stride, H, W):
#     min_len = min(len(events1), len(events2))
#     end = min_len - window + 1
#     mse_list = []

#     for start in range(0, end, stride):
#         slice1 = events1[start:start + window]
#         slice2 = events2[start:start + window]

#         cnt1 = events_to_count_map(slice1, H, W)
#         cnt2 = events_to_count_map(slice2, H, W)

#         mse = F.mse_loss(torch.tensor(cnt1), torch.tensor(cnt2)).item()
#         mse_list.append(mse)

#     return mse_list

# def main():
#     gt_events = load_events(gt_path)
#     pred_events = load_events(pred_path)

#     mse_list = compute_mse_for_all_windows(gt_events, pred_events, window, sliding_window, H, W)
#     mean_mse = np.mean(mse_list)

#     # 保存结果
#     with open(mse_save_path, 'w') as f:
#         f.write(f"Mean MSE: {mean_mse:.6f}\n")
#         for i, mse in enumerate(mse_list):
#             f.write(f"Window {i}: MSE = {mse:.6f}\n")

#     print(f"✅ MSE计算完成，结果保存在：{mse_save_path}")

# if __name__ == '__main__':
#     main()






import numpy as np
import os

def load_events(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 第一行是尺寸信息
    width, height = map(int, lines[0].strip().split())
    events = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        parts = list(map(int, line.strip().split()))
        if len(parts) == 4:
            events.append(parts)
    events = np.array(events)  # shape: [N, 4] -> t x y p
    return events, width, height

def split_events_into_N_parts(events, N):
    """根据时间戳划分为N段"""
    min_t, max_t = events[:, 0].min(), events[:, 0].max()
    time_range = max_t - min_t
    parts = []
    for i in range(N):
        t_start = min_t + i * time_range // N
        t_end = min_t + (i + 1) * time_range // N
        part = events[(events[:, 0] >= t_start) & (events[:, 0] < t_end)]
        parts.append(part)
    return parts

def generate_event_count_map(events, width, height):
    count_map = np.zeros((2, height, width), dtype=np.float32)
    for t, x, y, p in events:
        if 0 <= x < width and 0 <= y < height:
            count_map[p, y, x] += 1
    return count_map

def compute_mse(pred_maps, gt_maps):
    assert pred_maps.shape == gt_maps.shape
    return np.mean((pred_maps - gt_maps) ** 2, axis=(1, 2, 3))  # shape: [9]

def main():
    # 路径
    gt_path = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\dataset\SR_Test\HR\11\IR.txt"
    pred_path = r"C:\Users\chxu4146\Project\EvSR-result\NFS\7-26\baseline-50ms(guiyi)\HRPre\11\IR.txt"
    output_txt = os.path.join(os.path.dirname(pred_path), "mse.txt")

    N = 9  # 划分帧数

    # 加载事件
    gt_events, gt_w, gt_h = load_events(gt_path)
    pred_events, pred_w, pred_h = load_events(pred_path)
    assert (gt_w, gt_h) == (pred_w, pred_h), "尺寸不一致"

    # 分段
    gt_parts = split_events_into_N_parts(gt_events, N)
    pred_parts = split_events_into_N_parts(pred_events, N)

    # 构建计数图
    gt_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in gt_parts])  # [9, 2, H, W]
    pred_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in pred_parts])  # [9, 2, H, W]

    # 计算每帧MSE
    mse_per_frame = compute_mse(pred_maps, gt_maps)
    mean_mse = mse_per_frame.mean()

    # 保存
    with open(output_txt, 'w') as f:
        for i, mse in enumerate(mse_per_frame):
            f.write(f"Frame {i + 1}: MSE = {mse:.6f}\n")
        f.write(f"\nMean MSE (9 frames): {mean_mse:.6f}\n")

    print(f"[✔] Saved MSE to {output_txt}")

if __name__ == "__main__":
    main()
