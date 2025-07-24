import os
import numpy as np
import torch

def split_event_txt_by_time(input_txt, output_dir, time_window_us=50000):
    # 读取事件数据
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    # 读取第一行：图像尺寸
    try:
        width, height = map(int, lines[0].strip().split())
        print(f"[INFO] Sensor size: width={width}, height={height}")
    except:
        raise ValueError(f"文件 {input_txt} 的第一行格式应为 'width height'，实际为：{lines[0]}")

    # 读取事件行
    events = []
    for idx, line in enumerate(lines[1:]):
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                event = list(map(int, parts))
                events.append(event)
            except ValueError:
                print(f"[警告] 第 {idx+2} 行无法转换为整数，跳过：{line.strip()}")
        else:
            print(f"[警告] 第 {idx+2} 行格式错误，跳过：{line.strip()}")

    if len(events) == 0:
        print(f"[跳过] {input_txt} 无有效事件。")
        return

    # ✅ 使用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    events = torch.tensor(events, dtype=torch.int32, device=device)  # [N, 4]
    timestamps = events[:, 0]
    print(f"[INFO] {os.path.basename(input_txt)} 有效事件数：{len(events)}，处理设备：{device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 时间窗口划分
    start_time = int(timestamps.min().item())
    end_time   = int(timestamps.max().item())
    file_index = 1
    current_start = start_time
    current_end = current_start + time_window_us

    while current_start <= end_time:
        mask = (timestamps >= current_start) & (timestamps < current_end)
        window_events = events[mask]

        if window_events.shape[0] > 0:
            save_path = os.path.join(output_dir, f"{file_index}.npy")
            # ✅ 必须转回 CPU 再保存
            np.save(save_path, window_events.cpu().numpy())

        file_index += 1
        current_start = current_end
        current_end += time_window_us

    print(f"✅ {os.path.basename(input_txt)} 保存 {file_index - 1} 个窗口到 {output_dir}\n")


def batch_process_folder(input_dir, output_dir, time_window_us=50000):
    for filename in sorted(os.listdir(input_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if filename.endswith('.txt'):
            name = os.path.splitext(filename)[0]
            input_txt_path = os.path.join(input_dir, filename)
            output_npy_dir = os.path.join(output_dir, name)
            split_event_txt_by_time(input_txt_path, output_npy_dir, time_window_us)


# === 设置路径（替换为你的路径） ===
input_txt_folder = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\ev_hr"
output_npy_root  = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\HR"

# === 启动批量处理 ===
if __name__ == '__main__':
    batch_process_folder(input_txt_folder, output_npy_root, time_window_us=50000)
