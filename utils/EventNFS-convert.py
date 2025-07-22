import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_and_process_event_file(file_path, t_sample=1500):
    """
    读取单个 .txt，并把时间戳压缩到 [0, t_sample-1] 区间
    返回形状 (N, 4) 的 numpy 数组，列顺序 [t, x, y, p]
    """
    # 跳过首行分辨率
    data = pd.read_csv(file_path,
                       delim_whitespace=True,
                       header=None,
                       skiprows=1,
                       engine='c').values

    # 拆分字段
    t, x, y, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # 如果极性已经是 0/1，则无需转换
    p = p.astype(np.uint8)          # 可选：显式转无符号 8 位整数

    # 时间戳归一化到 [0, t_sample-1]
    t_min, t_max = t.min(), t.max()
    t_norm = ((t - t_min) / (t_max - t_min) * (t_sample - 1)).astype(np.int32)

    return np.stack([t_norm, x, y, p], axis=1)


def convert_txt_to_npy(input_dir, output_dir, t_sample=1500):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.txt'))

    for txt_file in tqdm(txt_files, desc=f"Processing {os.path.basename(input_dir)}"):
        src = os.path.join(input_dir, txt_file)
        arr = read_and_process_event_file(src, t_sample)
        dst = os.path.join(output_dir, txt_file.replace('.txt', '.npy'))
        np.save(dst, arr)


if __name__ == '__main__':
    # 路径配置
    hr_txt_dir = r"E:\EventSR-dataset\EventNFS\ev_hr"
    lr_txt_dir = r"E:\EventSR-dataset\EventNFS\ev_lr_1"
    hr_npy_dir = r"E:\EventSR-dataset\EventNFS\HR"
    lr_npy_dir = r"E:\EventSR-dataset\EventNFS\LR"

    # 执行转换
    convert_txt_to_npy(hr_txt_dir, hr_npy_dir, t_sample=1500)
    convert_txt_to_npy(lr_txt_dir, lr_npy_dir, t_sample=1500)
