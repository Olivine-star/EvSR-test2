# import numpy as np
# import os
# from skimage.metrics import structural_similarity as compare_ssim

# def load_events(txt_path):
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()

#     width, height = map(int, lines[0].strip().split())
#     events = []
#     for line in lines[1:]:
#         if line.strip() == "":
#             continue
#         parts = list(map(int, line.strip().split()))
#         if len(parts) == 4:
#             events.append(parts)
#     return np.array(events), width, height  # shape: [N, 4]

# def split_events_into_N_parts(events, N):
#     min_t, max_t = events[:, 0].min(), events[:, 0].max()
#     time_range = max_t - min_t
#     parts = []
#     for i in range(N):
#         t_start = min_t + i * time_range // N
#         t_end = min_t + (i + 1) * time_range // N
#         part = events[(events[:, 0] >= t_start) & (events[:, 0] < t_end)]
#         parts.append(part)
#     return parts

# def generate_event_count_map(events, width, height):
#     count_map = np.zeros((2, height, width), dtype=np.float32)
#     for t, x, y, p in events:
#         if 0 <= x < width and 0 <= y < height:
#             count_map[p, y, x] += 1
#     return count_map

# def compute_mse(pred_maps, gt_maps):
#     assert pred_maps.shape == gt_maps.shape
#     return np.mean((pred_maps - gt_maps) ** 2, axis=(1, 2, 3))  # shape: [N]

# def compute_rmse(pred_maps, gt_maps):
#     return np.sqrt(compute_mse(pred_maps, gt_maps))

# def compute_ssim(pred_maps, gt_maps):
#     # 返回每帧的平均 SSIM（两个通道的平均）
#     N = pred_maps.shape[0]
#     ssim_scores = []
#     for i in range(N):
#         ssim_0 = compare_ssim(pred_maps[i, 0], gt_maps[i, 0], data_range=gt_maps[i, 0].max() - gt_maps[i, 0].min())
#         ssim_1 = compare_ssim(pred_maps[i, 1], gt_maps[i, 1], data_range=gt_maps[i, 1].max() - gt_maps[i, 1].min())
#         ssim_scores.append((ssim_0 + ssim_1) / 2)
#     return np.array(ssim_scores)

# def process_pair(gt_path, pred_path, output_txt, N=9):
#     gt_events, gt_w, gt_h = load_events(gt_path)
#     pred_events, pred_w, pred_h = load_events(pred_path)

#     assert (gt_w, gt_h) == (pred_w, pred_h), f"尺寸不一致: {gt_path} vs {pred_path}"

#     gt_parts = split_events_into_N_parts(gt_events, N)
#     pred_parts = split_events_into_N_parts(pred_events, N)

#     gt_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in gt_parts])
#     pred_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in pred_parts])

#     mse_per_frame = compute_mse(pred_maps, gt_maps)
#     rmse_per_frame = np.sqrt(mse_per_frame)
#     ssim_per_frame = compute_ssim(pred_maps, gt_maps)

#     mean_mse = mse_per_frame.mean()
#     mean_rmse = rmse_per_frame.mean()
#     mean_ssim = ssim_per_frame.mean()

#     with open(output_txt, 'w') as f:
#         for i, (mse, rmse, ssim) in enumerate(zip(mse_per_frame, rmse_per_frame, ssim_per_frame)):
#             f.write(f"Frame {i + 1}: MSE = {mse:.6f}, RMSE = {rmse:.6f}, SSIM = {ssim:.6f}\n")
#         f.write(f"\nMean MSE (9 frames): {mean_mse:.6f}\n")
#         f.write(f"Mean RMSE (9 frames): {mean_rmse:.6f}\n")
#         f.write(f"Mean SSIM (9 frames): {mean_ssim:.6f}\n")

#     return mean_mse, mean_rmse, mean_ssim

# def main():
#     root_gt_dir = r"C:\code\EventSR-Project\EventSR-dataset\dataset\NFS\data-visual-test-redata-ef6\HR"
#     root_pred_dir = r"C:\code\EventSR-Project\EventSR-dataset\dataset\NFS\data-visual-test-redata-ef6\NFS-Louck-light-p-learn(50ms-redata)-leak"
#     summary_output = os.path.join(root_pred_dir, "mean-mse.txt")

#     all_stats = []

#     for folder in os.listdir(root_gt_dir):
#         gt_folder = os.path.join(root_gt_dir, folder)
#         pred_folder = os.path.join(root_pred_dir, folder)

#         gt_txt = os.path.join(gt_folder, "IR.txt")
#         pred_txt = os.path.join(pred_folder, "IR.txt")
#         mse_txt = os.path.join(pred_folder, "mse.txt")

#         if not os.path.exists(gt_txt) or not os.path.exists(pred_txt):
#             print(f"[跳过] 缺少 IR.txt 文件: {gt_txt} 或 {pred_txt}")
#             continue

#         mean_mse, mean_rmse, mean_ssim = process_pair(gt_txt, pred_txt, mse_txt)
#         all_stats.append((folder, mean_mse, mean_rmse, mean_ssim))
#         print(f"[✔] {folder}: MSE={mean_mse:.6f}, RMSE={mean_rmse:.6f}, SSIM={mean_ssim:.6f}")

#     # 保存整体统计
#     with open(summary_output, 'w') as f:
#         for folder, mse, rmse, ssim in all_stats:
#             f.write(f"{folder}: MSE = {mse:.6f}, RMSE = {rmse:.6f}, SSIM = {ssim:.6f}\n")

#         overall_mse = np.mean([m for _, m, _, _ in all_stats])
#         overall_rmse = np.mean([r for _, _, r, _ in all_stats])
#         overall_ssim = np.mean([s for _, _, _, s in all_stats])

#         f.write(f"\nOverall Mean MSE = {overall_mse:.6f}\n")
#         f.write(f"Overall Mean RMSE = {overall_rmse:.6f}\n")
#         f.write(f"Overall Mean SSIM = {overall_ssim:.6f}\n")

#     print(f"\n✅ 全部处理完成，结果保存于: {summary_output}")

# if __name__ == "__main__":
#     main()













import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn.functional as F

def load_events(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    width, height = map(int, lines[0].strip().split())
    events = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        parts = list(map(int, line.strip().split()))
        if len(parts) == 4:
            events.append(parts)
    return np.array(events), width, height  # shape: [N, 4]

def split_events_into_N_parts(events, N):
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
    # 转为 PyTorch tensor
    pred = torch.tensor(pred_maps)
    gt = torch.tensor(gt_maps)

    # 使用 MSELoss without reduction
    mse = F.mse_loss(pred, gt, reduction='none')  # shape: [N, 2, H, W]
    mse = mse.view(mse.size(0), -1).mean(dim=1)   # 每帧平均，shape: [N]
    return mse.numpy()  # 返回 numpy 以兼容原结构



def compute_rmse(pred_maps, gt_maps):
    return np.sqrt(compute_mse(pred_maps, gt_maps))

def compute_ssim(pred_maps, gt_maps):
    # 返回每帧的平均 SSIM（两个通道的平均）
    N = pred_maps.shape[0]
    ssim_scores = []
    for i in range(N):
        ssim_0 = compare_ssim(pred_maps[i, 0], gt_maps[i, 0], data_range=gt_maps[i, 0].max() - gt_maps[i, 0].min())
        ssim_1 = compare_ssim(pred_maps[i, 1], gt_maps[i, 1], data_range=gt_maps[i, 1].max() - gt_maps[i, 1].min())
        ssim_scores.append((ssim_0 + ssim_1) / 2)
    return np.array(ssim_scores)

def process_pair(gt_path, pred_path, output_txt, N=9):
    gt_events, gt_w, gt_h = load_events(gt_path)
    pred_events, pred_w, pred_h = load_events(pred_path)

    assert (gt_w, gt_h) == (pred_w, pred_h), f"尺寸不一致: {gt_path} vs {pred_path}"

    gt_parts = split_events_into_N_parts(gt_events, N)
    pred_parts = split_events_into_N_parts(pred_events, N)

    gt_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in gt_parts])
    pred_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in pred_parts])

    mse_per_frame = compute_mse(pred_maps, gt_maps)
    rmse_per_frame = np.sqrt(mse_per_frame)
    ssim_per_frame = compute_ssim(pred_maps, gt_maps)

    mean_mse = mse_per_frame.mean()
    mean_rmse = rmse_per_frame.mean()
    mean_ssim = ssim_per_frame.mean()

    with open(output_txt, 'w') as f:
        for i, (mse, rmse, ssim) in enumerate(zip(mse_per_frame, rmse_per_frame, ssim_per_frame)):
            f.write(f"Frame {i + 1}: MSE = {mse:.6f}, RMSE = {rmse:.6f}, SSIM = {ssim:.6f}\n")
        f.write(f"\nMean MSE (9 frames): {mean_mse:.6f}\n")
        f.write(f"Mean RMSE (9 frames): {mean_rmse:.6f}\n")
        f.write(f"Mean SSIM (9 frames): {mean_ssim:.6f}\n")

    return mean_mse, mean_rmse, mean_ssim

def main():
    root_gt_dir = r"C:\code\EventSR-Project\EventSR-dataset\dataset\NFS\data-visual-test-redata-ef6\HR"
    root_pred_dir = r"C:\code\EventSR-Project\EventSR-dataset\dataset\NFS\data-visual-test-redata-ef6\NFS-Louck-light-p(50ms-redata)-leak"
    summary_output = os.path.join(root_pred_dir, "mean-mse.txt")

    all_stats = []

    for folder in os.listdir(root_gt_dir):
        gt_folder = os.path.join(root_gt_dir, folder)
        pred_folder = os.path.join(root_pred_dir, folder)

        gt_txt = os.path.join(gt_folder, "IR.txt")
        pred_txt = os.path.join(pred_folder, "IR.txt")
        mse_txt = os.path.join(pred_folder, "mse.txt")

        if not os.path.exists(gt_txt) or not os.path.exists(pred_txt):
            print(f"[跳过] 缺少 IR.txt 文件: {gt_txt} 或 {pred_txt}")
            continue

        mean_mse, mean_rmse, mean_ssim = process_pair(gt_txt, pred_txt, mse_txt)
        all_stats.append((folder, mean_mse, mean_rmse, mean_ssim))
        print(f"[✔] {folder}: MSE={mean_mse:.6f}, RMSE={mean_rmse:.6f}, SSIM={mean_ssim:.6f}")

    # 保存整体统计
    with open(summary_output, 'w') as f:
        for folder, mse, rmse, ssim in all_stats:
            f.write(f"{folder}: MSE = {mse:.6f}, RMSE = {rmse:.6f}, SSIM = {ssim:.6f}\n")

        overall_mse = np.mean([m for _, m, _, _ in all_stats])
        overall_rmse = np.mean([r for _, _, r, _ in all_stats])
        overall_ssim = np.mean([s for _, _, _, s in all_stats])

        f.write(f"\nOverall Mean MSE = {overall_mse:.6f}\n")
        f.write(f"Overall Mean RMSE = {overall_rmse:.6f}\n")
        f.write(f"Overall Mean SSIM = {overall_ssim:.6f}\n")

    print(f"\n✅ 全部处理完成，结果保存于: {summary_output}")

if __name__ == "__main__":
    main()
