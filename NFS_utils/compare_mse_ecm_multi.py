import numpy as np
import os

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
    assert pred_maps.shape == gt_maps.shape
    return np.mean((pred_maps - gt_maps) ** 2, axis=(1, 2, 3))  # shape: [N]

def process_pair(gt_path, pred_path, output_txt, N=9):
    gt_events, gt_w, gt_h = load_events(gt_path)
    pred_events, pred_w, pred_h = load_events(pred_path)

    assert (gt_w, gt_h) == (pred_w, pred_h), f"尺寸不一致: {gt_path} vs {pred_path}"

    gt_parts = split_events_into_N_parts(gt_events, N)
    pred_parts = split_events_into_N_parts(pred_events, N)

    gt_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in gt_parts])
    pred_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in pred_parts])

    mse_per_frame = compute_mse(pred_maps, gt_maps)
    mean_mse = mse_per_frame.mean()

    with open(output_txt, 'w') as f:
        for i, mse in enumerate(mse_per_frame):
            f.write(f"Frame {i + 1}: MSE = {mse:.6f}\n")
        f.write(f"\nMean MSE (9 frames): {mean_mse:.6f}\n")

    return mean_mse

def main():
    root_gt_dir = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\dataset\SR_Test\HR"
    root_pred_dir = r"C:\Users\chxu4146\Project\EvSR-result\NFS\7-27\light-p-learn-50ms(guiyi)\HRPre"
    summary_output = os.path.join(root_pred_dir, "mean-mse.txt")

    all_means = []

    for folder in os.listdir(root_gt_dir):
        gt_folder = os.path.join(root_gt_dir, folder)
        pred_folder = os.path.join(root_pred_dir, folder)

        gt_txt = os.path.join(gt_folder, "IR.txt")
        pred_txt = os.path.join(pred_folder, "IR.txt")
        mse_txt = os.path.join(pred_folder, "mse.txt")

        if not os.path.exists(gt_txt) or not os.path.exists(pred_txt):
            print(f"[跳过] 缺少 IR.txt 文件: {gt_txt} 或 {pred_txt}")
            continue

        mean_mse = process_pair(gt_txt, pred_txt, mse_txt)
        all_means.append((folder, mean_mse))
        print(f"[✔] {folder}: Mean MSE = {mean_mse:.6f}")

    # 保存所有类别的平均 MSE
    with open(summary_output, 'w') as f:
        for folder, mse in all_means:
            f.write(f"{folder}: Mean MSE = {mse:.6f}\n")
        mean_all = np.mean([mse for _, mse in all_means])
        f.write(f"\nOverall Mean MSE = {mean_all:.6f}\n")

    print(f"\n✅ 全部处理完成，结果保存于: {summary_output}")

if __name__ == "__main__":
    main()
