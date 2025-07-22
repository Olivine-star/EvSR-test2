import os
import random
import shutil
from pathlib import Path

# 原始数据根目录
ROOT = Path(r"E:\EventSR-dataset\EventNFS")
HR_DIR = ROOT / "HR"
LR_DIR = ROOT / "LR"

# 目标输出目录
HR_TRAIN = ROOT / "HR_train"
LR_TRAIN = ROOT / "LR_train"
HR_TEST  = ROOT / "HR_test"
LR_TEST  = ROOT / "LR_test"

# ① 创建目标文件夹（存在则跳过）
for d in [HR_TRAIN, LR_TRAIN, HR_TEST, LR_TEST]:
    d.mkdir(parents=True, exist_ok=True)

# ② 获取 HR / LR 对应文件名（假设都是纯数字+.npy）
hr_files = sorted([f for f in HR_DIR.glob("*.npy") if f.stem.isdigit()])
lr_files = sorted([f for f in LR_DIR.glob("*.npy") if f.stem.isdigit()])

# ③ 简单一致性检查
hr_ids = {f.stem for f in hr_files}
lr_ids = {f.stem for f in lr_files}
if hr_ids != lr_ids:
    missing_hr = lr_ids - hr_ids
    missing_lr = hr_ids - lr_ids
    raise ValueError(
        f"HR/LR 文件编号不一致！\n"
        f"   HR 缺少: {sorted(missing_hr)}\n"
        f"   LR 缺少: {sorted(missing_lr)}"
    )

all_ids = sorted(list(hr_ids))  # 字符串数字，已经相同

# ④ 随机抽取 20% 作为测试集
random.seed(42)                    # 固定随机种子，保证可复现
num_test = max(1, int(len(all_ids) * 0.2 + 0.5))  # 四舍五入
test_ids = set(random.sample(all_ids, num_test))

# ⑤ 按照 id 拷贝 / 移动文件
def split_and_copy(id_set: set, src_dir: Path, dst_train: Path, dst_test: Path):
    """把 src_dir 里的文件按 id_set 分到 train/test"""
    for f in src_dir.glob("*.npy"):
        target_dir = dst_test if f.stem in id_set else dst_train
        shutil.copy2(f, target_dir / f.name)         # 如需“移动”改成 shutil.move
        # shutil.move(f, target_dir / f.name)        # ←改行即可把原文件移走

split_and_copy(test_ids, HR_DIR, HR_TRAIN, HR_TEST)
split_and_copy(test_ids, LR_DIR, LR_TRAIN, LR_TEST)

print(
    f"完成划分：\n"
    f"  训练集: {len(all_ids) - num_test} 对 → {HR_TRAIN} / {LR_TRAIN}\n"
    f"  测试集: {num_test} 对 → {HR_TEST} / {LR_TEST}"
)
