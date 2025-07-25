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

    # === ✅ 索引合法性检查 ===
    assert tGt.max() < _T and tGt.min() >= 0, f"tGt out of range: min={tGt.min()}, max={tGt.max()}, expected [0, {_T-1}]"
    assert xGt.max() < _H and xGt.min() >= 0, f"xGt out of range: min={xGt.min()}, max={xGt.max()}, expected [0, {_H-1}]"
    assert yGt.max() < _W and yGt.min() >= 0, f"yGt out of range: min={yGt.min()}, max={yGt.max()}, expected [0, {_W-1}]"
    assert pGt.max() <= 1 and pGt.min() >= 0, f"pGt out of range: unique={torch.unique(pGt)}"

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
_H, _W, _T = [223, 125, 50]

def running_cal_RMSE(savepath, sr_test_root):
    path1, path2 = savepath, sr_test_root
    #_H, _W, _T = [240, 180, 600]

    classList = os.listdir(os.path.join(path2, 'HR'))

    # === 初始化列表 ===
    RMSEListOurs, RMSEListOurs_s, RMSEListOurs_t = [], [], []
    PAList = []  # 极性准确率列表

    i = 1
    for n in classList:
        print(f"Class: {n}")
        p1 = os.path.join(path1, n)              # Output
        p2 = os.path.join(path2, 'HR', n)        # GT
        sampleList = os.listdir(p2)

        for k, name in enumerate(sampleList, 1):
            path_out = os.path.join(p1, name)
            path_gt  = os.path.join(p2, name)
            if not os.path.exists(path_out) or not os.path.exists(path_gt):
                print(f"🚫 Missing file: {name}, skipping...")
                continue

            eventOutput = np.load(path_out)
            eventGt     = np.load(path_gt)


            RMSE, RMSE_t, RMSE_s, PA = calRMSE(eventOutput, eventGt)

            RMSEListOurs.append(RMSE)
            RMSEListOurs_s.append(RMSE_s)
            RMSEListOurs_t.append(RMSE_t)
            PAList.append(PA)

            print(f"{i}/{len(classList)}   {k}/{len(sampleList)}  RMSE: {RMSE:.4f}  RMSE_t: {RMSE_t:.4f}  RMSE_s: {RMSE_s:.4f}  Polarity Accuracy: {PA:.4f}")
        i += 1

    # === 汇总结果 ===
    rmse_mean = sum(RMSEListOurs) / len(RMSEListOurs)
    rmse_s_mean = sum(RMSEListOurs_s) / len(RMSEListOurs)
    rmse_t_mean = sum(RMSEListOurs_t) / len(RMSEListOurs)
    pa_mean = sum(PAList) / len(PAList)

    # === 写入报告 ===
    result_path = os.path.join(path1, 'result.txt')
    with open(result_path, 'w') as f:
        f.write("==== Event-Based SR Evaluation Result ====\n")
        f.write(f"Total Samples: {len(RMSEListOurs)}\n")
        f.write(f"RMSE (total): {rmse_mean:.4f}\n")
        f.write(f"RMSE (spatial): {rmse_s_mean:.4f}\n")
        f.write(f"RMSE (temporal): {rmse_t_mean:.4f}\n")
        f.write(f"Polarity Accuracy: {pa_mean:.4f}\n")
        f.write("==========================================\n")

    return rmse_mean, rmse_s_mean, rmse_t_mean, pa_mean

    print("\n✅ Evaluation Summary")
    print(f"Total Samples     : {len(RMSEListOurs)}")
    print(f"RMSE (total)      : {rmse_mean:.4f}")
    print(f"RMSE (spatial)    : {rmse_s_mean:.4f}")
    print(f"RMSE (temporal)   : {rmse_t_mean:.4f}")
    print(f"Polarity Accuracy : {pa_mean:.4f}")
    print(f"📄 Results written to: {result_path}")


if __name__ == '__main__':
    # === 路径读取逻辑保持不变 ===
    def load_path_config(path_config='nfs_path.txt'):
        path_dict = {}
        with open(path_config, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=', 1)
                    path_dict[key.strip()] = val.strip()
        return path_dict

    paths = load_path_config()
    path1 = paths.get('savepath', '')           # Output
    path2 = paths.get('sr_test_root', '')       # GT
    running_cal_RMSE(path1, path2)