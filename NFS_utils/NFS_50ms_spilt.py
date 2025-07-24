import numpy as np
import os

def split_event_txt_by_time(input_txt, output_dir, time_window_us=50000):
    # 读取事件数据
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    
    # 读取第一行：图像尺寸（忽略后续处理，但可以打印验证）
    try:
        width, height = map(int, lines[0].strip().split())
        print(f"Sensor size: width={width}, height={height}")
    except:
        raise ValueError("第一行必须是图像尺寸，如 '240 180'")

    # 读取有效事件行：每行应包含 t x y p 四个整数
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

    events = np.array(events, dtype=np.int32)
    print(f"总共读取有效事件数：{len(events)}")

    if len(events) == 0:
        print("没有可用事件，终止处理。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取时间范围
    start_time = events[:, 0].min()
    end_time = events[:, 0].max()

    # 分时间窗口保存事件
    file_index = 1
    current_start = start_time
    current_end = current_start + time_window_us

    while current_start <= end_time:
        mask = (events[:, 0] >= current_start) & (events[:, 0] < current_end)
        window_events = events[mask]

        if len(window_events) > 0:
            np.save(os.path.join(output_dir, f"{file_index}.npy"), window_events)

        file_index += 1
        current_start = current_end
        current_end += time_window_us

    print(f"✅ 成功保存 {file_index - 1} 个窗口事件到: {output_dir}")


# === 输入路径和输出路径 ===
input_txt_path = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\ev_hr\1.txt"
output_npy_dir = r"C:\Users\chxu4146\Project\EvSR-dataset\NFS\HR\1"

# === 执行分割 ===
if __name__ == '__main__':
    split_event_txt_by_time(input_txt_path, output_npy_dir, time_window_us=50000)
