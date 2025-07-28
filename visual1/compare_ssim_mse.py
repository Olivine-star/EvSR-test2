import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error

# ✅ 输入路径
output_img_path = "output.png"    # 你的模型重建结果图像
gt_img_path = "GT.png"            # Ground Truth 图像

# ✅ 输出结果文件夹
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
result_file = os.path.join(output_dir, "result.txt")

# ✅ 加载图像（灰度）
img1 = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)

# ✅ 图像尺寸检查
if img1.shape != img2.shape:
    raise ValueError(f"[错误] 图像尺寸不一致: {img1.shape} vs {img2.shape}")

# ✅ SSIM 计算
ssim_score, _ = compare_ssim(img1, img2, full=True)

# ✅ MSE 计算
mse_score = mean_squared_error(img1.flatten(), img2.flatten())

# ✅ 打印结果
result_text = f"SSIM: {ssim_score:.4f}\nMSE: {mse_score:.4f}"
print(result_text)

# ✅ 写入文件
with open(result_file, "w") as f:
    f.write(result_text)

print(f"[INFO] 比较结果已保存到: {result_file}")
