import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# 参数设置
input_dir = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\HR-npy"
output_dir = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\HR-gif"
num_events_per_frame = 10000
H, W = 260, 346  # 分辨率

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历 1.npy 到 100.npy
for idx in range(1, 101):
    filename = f"{idx}.npy"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{idx}.gif")

    if not os.path.exists(input_path):
        print(f"[跳过] 文件不存在: {input_path}")
        continue

    # 读取事件数据
    events = np.load(input_path)  # shape: [N, 4], [t, x, y, p]
    frames = []

    num_frames = len(events) // num_events_per_frame
    for i in range(num_frames):
        frame_events = events[i * num_events_per_frame : (i + 1) * num_events_per_frame]
        frame_img = np.zeros((H, W), dtype=np.uint8)

        for event in frame_events:
            t, x, y, p = event.astype(int)
            if 0 <= x < W and 0 <= y < H:
                frame_img[y, x] = 255 if p == 1 else 100

        rgb_frame = np.stack([frame_img]*3, axis=-1)
        frames.append(rgb_frame)

    if frames:
        imageio.mimsave(output_path, frames, duration=0.1)  # 每帧间隔 0.1s
        print(f"[完成] 保存 GIF: {output_path}")
    else:
        print(f"[跳过] 无帧可生成: {input_path}")
