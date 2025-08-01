import os
import numpy as np
from tqdm import tqdm

def center_crop(events, crop_size=32, original_size=34):
    """
    裁剪事件的空间坐标到中心区域
    """
    margin = (original_size - crop_size) // 2  # == 1 for 34 -> 32
    x, y = events[:, 1], events[:, 2]
    mask = (x >= margin) & (x < original_size - margin) & \
           (y >= margin) & (y < original_size - margin)
    events = events[mask]
    events[:, 1] -= margin
    events[:, 2] -= margin
    return events

# 原始和输出路径
input_dirs = [
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\HR",
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR"
]
output_dirs = [
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\HR-32",
    r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR-32"
]

for input_root, output_root in zip(input_dirs, output_dirs):
    os.makedirs(output_root, exist_ok=True)
    
    for class_name in os.listdir(input_root):
        input_class_dir = os.path.join(input_root, class_name)
        output_class_dir = os.path.join(output_root, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for fname in tqdm(os.listdir(input_class_dir), desc=f"Cropping {class_name}"):
            if not fname.endswith('.npy'):
                continue
            input_path = os.path.join(input_class_dir, fname)
            output_path = os.path.join(output_class_dir, fname)

            events = np.load(input_path)  # shape: [N, 4]
            cropped = center_crop(events, crop_size=32, original_size=34)
            np.save(output_path, cropped.astype(np.int16))
