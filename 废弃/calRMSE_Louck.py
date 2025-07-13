import numpy as np
import os
import math

def calRMSE(eventOutput, eventGt):
    xOp = np.round(eventOutput[:, 1]).astype(int)
    yOp = np.round(eventOutput[:, 2]).astype(int)
    pOp = np.round(eventOutput[:, 3]).astype(int)
    tOp = np.round(eventOutput[:, 0]).astype(int)

    xGt = np.round(eventGt[:, 1]).astype(int)
    yGt = np.round(eventGt[:, 2]).astype(int)
    pGt = np.round(eventGt[:, 3]).astype(int)
    tGt = np.round(eventGt[:, 0]).astype(int)

    # 构建字典索引 (t, x, y) → polarity
    pred_dict = {(t, x, y): p for t, x, y, p in zip(tOp, xOp, yOp, pOp)}
    gt_dict   = {(t, x, y): p for t, x, y, p in zip(tGt, xGt, yGt, pGt)}

    correct_match = 0
    common_coords = set(pred_dict.keys()).intersection(gt_dict.keys())
    for key in common_coords:
        if pred_dict[key] == gt_dict[key]:
            correct_match += 1
    polarity_acc = correct_match / len(common_coords) if common_coords else 0

    # 再跑原有 RMSE 逻辑
    VoxOp = np.zeros([2, _H, _W, _T])
    VoxOp[pOp, xOp, yOp, tOp] = 1
    VoxGt = np.zeros([2, _H, _W, _T])
    VoxGt[pGt, xGt, yGt, tGt] = 1
    ecm = np.sum(np.sum(VoxGt, axis=3), axis=0)



    RMSE1 = np.sum((VoxGt - VoxOp) ** 2)

    RMSE2 = 0
    for k in range(math.ceil(_T / 50)):
        psthGt = np.sum(VoxGt[:, :, :, k * 50:(k + 1) * 50], axis=3)
        psthOp = np.sum(VoxOp[:, :, :, k * 50:(k + 1) * 50], axis=3)
        RMSE2 += np.sum((psthGt - psthOp) ** 2)

    denom = (tGt.max() - tGt.min()) * np.sum(ecm != 0)
    print(np.sum(ecm != 0))

    RMSE = np.sqrt((RMSE1 + RMSE2) / denom)

    print(denom)
    RMSE_s = np.sqrt(RMSE1 / denom)
    RMSE_t = np.sqrt(RMSE2 / denom)

    return RMSE, RMSE_s, RMSE_t, polarity_acc


# === 路径读取逻辑保持不变 ===
def load_path_config(path_config='dataset_path.txt'):
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

_H, _W, _T = [240, 180, 600]

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
        eventOutput = np.load(os.path.join(p1, name))
        eventGt = np.load(os.path.join(p2, name))

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

print(f"\n✅ Final RMSE: {rmse_mean:.4f}  | Polarity Acc: {pa_mean:.3f}")
print(f"📄 Results written to: {result_path}")
