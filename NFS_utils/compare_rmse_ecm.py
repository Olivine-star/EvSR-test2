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
    events = np.array(events)  # [N, 4]
    return events, width, height

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

def compute_rmse(pred_maps, gt_maps):
    assert pred_maps.shape == gt_maps.shape
    mse = np.mean((pred_maps - gt_maps) ** 2, axis=(1, 2, 3))  # [9]
    rmse = np.sqrt(mse)
    return rmse

def main():
    gt_path = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\dataset\SR_Test\HR\2\IR.txt"
    pred_path = r"C:\Users\chxu4146\Project\EvSR-result\NFS\7-26\baseline-50ms(guiyi)\HRPre\2\IR.txt"
    output_txt = os.path.join(os.path.dirname(pred_path), "rmse.txt")

    N = 9

    gt_events, gt_w, gt_h = load_events(gt_path)
    pred_events, pred_w, pred_h = load_events(pred_path)
    assert (gt_w, gt_h) == (pred_w, pred_h), "尺寸不一致"

    gt_parts = split_events_into_N_parts(gt_events, N)
    pred_parts = split_events_into_N_parts(pred_events, N)

    gt_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in gt_parts])
    pred_maps = np.stack([generate_event_count_map(e, gt_w, gt_h) for e in pred_parts])

    rmse_per_frame = compute_rmse(pred_maps, gt_maps)
    mean_rmse = rmse_per_frame.mean()

    with open(output_txt, 'w') as f:
        for i, rmse in enumerate(rmse_per_frame):
            f.write(f"Frame {i + 1}: RMSE = {rmse:.6f}\n")
        f.write(f"\nMean RMSE (9 frames): {mean_rmse:.6f}\n")

    print(f"[✔] Saved RMSE to {output_txt}")

if __name__ == "__main__":
    main()
