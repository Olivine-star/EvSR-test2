import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# 参数设置
event_path = r"E:\EventSR-dataset\NFS\dataset\SR_Train\HR\1.npy"
events = np.load(event_path)  # shape: [N, 4], columns = [t, x, y, p]
num_events_per_frame = 100_000
H, W = 260, 346  # 修改为你数据的分辨率
save_gif_path = "event_frames2.gif"

# 存储每帧图像
frames = []

# 遍历分帧
num_frames = len(events) // num_events_per_frame
for i in range(num_frames):
    # 获取当前帧的事件
    frame_events = events[i * num_events_per_frame : (i + 1) * num_events_per_frame]

    # 创建空白图像（灰度图）
    frame_img = np.zeros((H, W), dtype=np.uint8)

    # 渲染事件：正极性为白色(255)，负极性为灰色(100)
    for event in frame_events:
        t, x, y, p = event.astype(int)
        if 0 <= x < W and 0 <= y < H:
            frame_img[y, x] = 255 if p == 1 else 100

    # 转为RGB用于GIF保存
    rgb_frame = np.stack([frame_img]*3, axis=-1)
    frames.append(rgb_frame)

# 保存为GIF
imageio.mimsave(save_gif_path, frames, duration=1)  # 每帧间隔 0.1s

print(f"保存成功！GIF 路径: {save_gif_path}")
