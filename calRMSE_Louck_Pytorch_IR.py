import numpy as np
import os
import math
import numpy
import torch

def calRMSE(eventOutput, eventGt, device='cuda'):
    """
    使用 PyTorch 加速 RMSE 与极性准确率计算（兼容 torch<2.1，无需 torch.intersect1d）。
    输入:
        eventOutput, eventGt: numpy arrays, shape [N, 4], columns = [t, x, y, p]
    输出:
        RMSE, RMSE_s, RMSE_t, polarity_accuracy
    """
    eventOutput = torch.tensor(eventOutput, dtype=torch.long, device=device)
    eventGt = torch.tensor(eventGt, dtype=torch.long, device=device)


    # ✅ 加在这儿
# === 🔧 时间归一化到 [0, _T-1] 范围 ===
    t_all = torch.cat([eventGt[:, 0], eventOutput[:, 0]])
    t_min = t_all.min()
    t_max = t_all.max()
    t_range = t_max - t_min
    eventGt[:, 0] = ((eventGt[:, 0] - t_min) * (_T - 1) / t_range).long()
    eventOutput[:, 0] = ((eventOutput[:, 0] - t_min) * (_T - 1) / t_range).long()

    tOp, xOp, yOp, pOp = eventOutput[:, 0], eventOutput[:, 1], eventOutput[:, 2], eventOutput[:, 3]
    tGt, xGt, yGt, pGt = eventGt[:, 0], eventGt[:, 1], eventGt[:, 2], eventGt[:, 3]

    # 构建体素表示
    VoxOp = torch.zeros((2, _H, _W, _T), dtype=torch.float32, device=device)
    VoxGt = torch.zeros((2, _H, _W, _T), dtype=torch.float32, device=device)

    # 🔍 在赋值前打印索引范围
    print("tGt min/max:", tGt.min().item(), tGt.max().item())
    print("xGt min/max:", xGt.min().item(), xGt.max().item())
    print("yGt min/max:", yGt.min().item(), yGt.max().item())
    print("pGt unique:", torch.unique(pGt))
    print("Vox shape:", VoxGt.shape)

    VoxOp[pOp, xOp, yOp, tOp] = 1
    VoxGt[pGt, xGt, yGt, tGt] = 1


    # RMSE1: 全体素误差
    RMSE1 = torch.sum((VoxGt - VoxOp) ** 2)


    # RMSE2: 每 50 帧的时间块误差
    RMSE2 = 0
    block_size = 50
    for k in range((_T + block_size - 1) // block_size):
        t_start, t_end = k * block_size, min((k + 1) * block_size, _T)
        psthGt = torch.sum(VoxGt[:, :, :, t_start:t_end], dim=3)
        psthOp = torch.sum(VoxOp[:, :, :, t_start:t_end], dim=3)
        RMSE2 += torch.sum((psthGt - psthOp) ** 2)

    # === 极性准确率 ===
    # 唯一标识每个时空点 (t, x, y)
    flatOp = (tOp * _H * _W + xOp * _W + yOp).cpu().numpy()
    flatGt = (tGt * _H * _W + xGt * _W + yGt).cpu().numpy()

    # 使用 numpy 计算共同索引
    common_coords, idx_op_np, idx_gt_np = np.intersect1d(flatOp, flatGt, return_indices=True)

    if len(common_coords) > 0:
        correct_match = (pOp.cpu().numpy()[idx_op_np] == pGt.cpu().numpy()[idx_gt_np]).sum()
        polarity_acc = correct_match / len(common_coords)
    else:
        polarity_acc = 0.0

    # 归一化因子
    active_voxels = torch.sum(torch.sum(torch.sum(VoxGt, dim=3), dim=0) != 0)
    time_span = (tGt.max() - tGt.min()).item()
    denom = time_span * active_voxels.item()

    #denom = (tGt.max() - tGt.min()).item() * (torch.sum(torch.sum(VoxGt, dim=3) != 0)).item()
    #denom = (tGt.max() - tGt.min()).item() * np.sum(torch.sum(VoxGt, dim=3).cpu().numpy() != 0)

    #print((torch.sum(torch.sum(VoxGt, dim=3) != 0)).item())
    #ecm = torch.sum(VoxGt, dim=3)  # [2, H, W]
    #active_voxels = torch.sum(ecm != 0)  # 统计非零位置数
    #time_span = (tGt.max() - tGt.min()).item()  # 时间跨度
    #denom = time_span * active_voxels.item()

    RMSE = torch.sqrt((RMSE1 + RMSE2) / denom)
    #print(denom)
    RMSE_s = torch.sqrt(RMSE1 / denom)
    RMSE_t = torch.sqrt(RMSE2 / denom)

    return RMSE.item(), RMSE_s.item(), RMSE_t.item(), polarity_acc




# IR
_H, _W, _T = [240, 180, 50]

def evaluate_category(cat_dir, device='cuda'):
    """cat_dir 形如  C:/.../test/boxes_6dof"""
    pred_dir = os.path.join(cat_dir, 'SR_Test', 'HRPre2')
    gt_dir   = os.path.join(cat_dir, 'SR_Test', 'HR')

    if not (os.path.isdir(pred_dir) and os.path.isdir(gt_dir)):
        print(f"⚠️  跳过 {cat_dir}（缺少 HR 或 HRPre2）")
        return

    # 只评估共同存在的文件
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith('.npy')}
    gt_files   = {f for f in os.listdir(gt_dir)   if f.endswith('.npy')}
    common = sorted(pred_files & gt_files, key=lambda x: int(os.path.splitext(x)[0]))

    if not common:
        print(f"⚠️  {cat_dir} 没有共同的 .npy 文件可评估")
        return

    rmse_all, rmse_s_all, rmse_t_all, pa_all = [], [], [], []

    for idx, fname in enumerate(common, 1):
        out = np.load(os.path.join(pred_dir, fname))
        gt  = np.load(os.path.join(gt_dir,  fname))

        rmse, rmse_s, rmse_t, pa = calRMSE(out, gt, device=device)
        rmse_all.append(rmse)
        rmse_s_all.append(rmse_s)
        rmse_t_all.append(rmse_t)
        pa_all.append(pa)

        print(f"[{os.path.basename(cat_dir)}] {idx}/{len(common)}  "
              f"{fname:>6}  RMSE={rmse:.4f}  Rs={rmse_s:.4f}  Rt={rmse_t:.4f}  PA={pa:.4f}")

    # ----- 汇总并写 result.txt -----
    def mean(lst): return sum(lst) / len(lst)

    result_text = (
        "==== Event Based SR Evaluation Result ====\n"
        f"Category          : {os.path.basename(cat_dir)}\n"
        f"Samples Evaluated : {len(common)}\n"
        f"RMSE (total)      : {mean(rmse_all):.4f}\n"
        f"RMSE (spatial)    : {mean(rmse_s_all):.4f}\n"
        f"RMSE (temporal)   : {mean(rmse_t_all):.4f}\n"
        f"Polarity Accuracy : {mean(pa_all):.4f}\n"
        "==========================================\n"
    )

    save_path = os.path.join(pred_dir, 'result.txt')
    # with open(save_path, 'w') as f:
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(result_text)

    print(f"✅  结果已保存到 {save_path}\n")


def load_path_config(cfg_file='IR_evl.txt'):
    d = {}
    with open(cfg_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                d[k.strip()] = v.strip()
    return d


if __name__ == '__main__':
    cfg = load_path_config()  # 默认读取当前目录下 dataset_path.txt
    test_root = cfg.get('test_root', '')
    device    = cfg.get('device', 'cuda')

    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"测试根目录不存在：{test_root}")

    # 遍历所有类别文件夹
    for cat in sorted(os.listdir(test_root)):
        cat_dir = os.path.join(test_root, cat)
        if os.path.isdir(cat_dir):
            evaluate_category(cat_dir, device=device)
