import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_pixel_event_distribution(events, img_size=(64, 64), polarity_filter='both'):
    pixel_event_counts = np.zeros(img_size, dtype=int)

    for event in events:
        ts, x, y, p = event
        if polarity_filter == 'both' or (polarity_filter == 'positive' and p > 0) or (polarity_filter == 'negative' and p <= 0):
            pixel_event_counts[int(y), int(x)] += 1

    return pixel_event_counts


def plot_event_comparison_three_way(baseline_root, our_root, gt_root, category_id, sample_id, polarity_filter='both'):
    # 拼接路径
    baseline_file = os.path.join(baseline_root, str(category_id), f"{sample_id}.npy")
    our_file = os.path.join(our_root, str(category_id), f"{sample_id}.npy")
    gt_file = os.path.join(gt_root, str(category_id), f"{sample_id}.npy")

    # 检查文件是否存在-
    if not os.path.exists(baseline_file):
        print(f"❌ baseline 文件不存在: {baseline_file}")
        return
    if not os.path.exists(our_file):
        print(f"❌ our 文件不存在: {our_file}")
        return
    if not os.path.exists(gt_file):
        print(f"❌ ground truth 文件不存在: {gt_file}")
        return

    # 加载事件数据
    baseline_events = np.load(baseline_file)
    our_events = np.load(our_file)
    gt_events = np.load(gt_file)

    # 计算事件图-
    baseline_counts = analyze_pixel_event_distribution(baseline_events, polarity_filter=polarity_filter)
    our_counts = analyze_pixel_event_distribution(our_events, polarity_filter=polarity_filter)
    gt_counts = analyze_pixel_event_distribution(gt_events, polarity_filter=polarity_filter)

    # 颜色映射-
    if polarity_filter == 'both':
        cmap = 'Purples'
    elif polarity_filter == 'positive':
        cmap = 'Reds'
    elif polarity_filter == 'negative':
        cmap = 'Blues'

    # 可视化三张图
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(baseline_counts, cmap=cmap)
    plt.title('Baseline HRPre', fontsize=16)
    plt.xlabel("x [pixels]", fontsize=12)
    plt.ylabel("y [pixels]", fontsize=12)
    plt.colorbar(label='Number of Events')

    plt.subplot(1, 3, 2)
    plt.imshow(our_counts, cmap=cmap)
    plt.title('Our Model HRPre', fontsize=16)
    plt.xlabel("x [pixels]", fontsize=12)
    plt.ylabel("y [pixels]", fontsize=12)
    plt.colorbar(label='Number of Events')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_counts, cmap=cmap)
    plt.title('Ground Truth', fontsize=16)
    plt.xlabel("x [pixels]", fontsize=12)
    plt.ylabel("y [pixels]", fontsize=12)
    plt.colorbar(label='Number of Events')

    plt.suptitle(f'Category {category_id}, Sample {sample_id} — Polarity: {polarity_filter}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# =============================
# 示例调用
# =============================
baseline_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\ResConv\HRPre"
our_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\ResConv-attention\HRPre"
gt_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR"

category = 0   # 类别
sample = 0     # 样本编号

# 可选参数 polarity_filter: 'both', 'positive', 'negative'

plot_event_comparison_three_way(baseline_path, our_path, gt_path, category, sample, polarity_filter='both')
