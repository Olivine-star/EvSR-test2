import os
from nMnist.testNmnist_Louck import inference
from calRMSE_Louck_Pytorch import running_cal_RMSE

# -------------------------------
# ✅ 路径读取函数
# -------------------------------
def load_path_config(path_config='dataset_path.txt'):
    path_dict = {}
    with open(path_config, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                path_dict[key.strip()] = val.strip()
    return path_dict

# -------------------------------
# ✅ 入口主程序
# -------------------------------
def running_inference(ckpt_number=10, ckpt_root=None, hrPath=None, lrPath=None, base_savepath=None):
    paths = load_path_config('dataset_path.txt')
    if ckpt_root is None:
        ckpt_root = paths.get('ckptPath', '')
    if hrPath is None:
        hrPath = paths.get('test_hr', '')
    if lrPath is None:
        lrPath = paths.get('test_lr', '')
    if base_savepath is None:
        base_savepath = paths.get('savepath', '')


    logfile = os.path.join(base_savepath, 'all_ckpt_results.txt')
    os.makedirs(base_savepath, exist_ok=True)

    best_ckpt = None
    best_rmse = float('inf')
    results_all = []

    for i in range(ckpt_number):  # ckpt0 到 ckpt10
        ckpt_name = f'ckpt{i}'
        savepath = os.path.join(base_savepath, ckpt_name)
        os.makedirs(savepath, exist_ok=True)

        print(f"\n▶️  Running inference for {ckpt_name} ...")
        inference(ckpt_root, hrPath, lrPath, savepath, ckptname=ckpt_name)

        print(f"📊 Evaluating {ckpt_name} ...")
        rmse, rmse_s, rmse_t, pa = running_cal_RMSE(savepath, os.path.dirname(hrPath))

        results_all.append((ckpt_name, rmse, rmse_s, rmse_t, pa))

        if rmse < best_rmse:
            best_rmse = rmse
            best_ckpt = ckpt_name

    # === 写入日志文件 ===
    with open(logfile, 'w') as f:
        f.write("===== All Checkpoint Results =====\n")
        for ckpt_name, rmse, rmse_s, rmse_t, pa in results_all:
            flag = ' <-- Best' if ckpt_name == best_ckpt else ''
            f.write(f"{ckpt_name}: RMSE={rmse:.4f}, RMSE_s={rmse_s:.4f}, RMSE_t={rmse_t:.4f}, PA={pa:.4f}{flag}\n")
        f.write("==================================\n")
        f.write(f"✅ Best Checkpoint: {best_ckpt} with RMSE={best_rmse:.4f}\n")

    print(f"\n✅ All results written to {logfile}")
    print(f"🏆 Best checkpoint: {best_ckpt}")

if __name__ == '__main__':
    running_inference()
