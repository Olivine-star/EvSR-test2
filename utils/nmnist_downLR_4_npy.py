import os
import numpy as np
from tqdm import tqdm

# 原始和保存路径
input_root_dirs = [
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR",
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\LR"
]
output_root_dirs = [
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR-low",
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\LR-low"
]

# 遍历两个集合（train/test）
for input_root, output_root in zip(input_root_dirs, output_root_dirs):
    os.makedirs(output_root, exist_ok=True)

    for class_name in os.listdir(input_root):
        class_input_dir = os.path.join(input_root, class_name)
        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for file_name in tqdm(os.listdir(class_input_dir), desc=f"Processing {class_input_dir}"):
            if not file_name.endswith('.npy'):
                continue

            input_path = os.path.join(class_input_dir, file_name)
            output_path = os.path.join(class_output_dir, file_name)

            events = np.load(input_path)  # shape: [N, 4], columns: t, x, y, p
            downsampled = []

            for e in events:
                t, x, y, p = e
                if x >= 16 or y >= 16:  # 忽略无法整除的边缘
                    continue
                x_new = x // 2
                y_new = y // 2
                downsampled.append([t, x_new, y_new, p])

            downsampled = np.array(downsampled, dtype=np.int16)
            np.save(output_path, downsampled)
