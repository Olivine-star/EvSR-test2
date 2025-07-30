import os
import cv2
import numpy as np
import glob

# ✅ 所有路径和前缀
folders = {
    "GT": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\images",
    "HR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HR\E2VID_Results",
    "LR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\LR\E2VID_Results",
    "BS": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre\E2VID_Results",
    "light": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light\E2VID_Results",
    "light-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light-p-learn\E2VID_Results",
    "Louck-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-Louck-light-p\E2VID_Results",
    "Louck-l-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre2(Louck-light-p-learn)\E2VID_Results"
}

# ✅ 获取所有图片列表并排序
images_dict = {}
for prefix, path in folders.items():
    img_list = sorted(glob.glob(os.path.join(path, "*.png")))
    images_dict[prefix] = img_list

# ✅ 统一对比图数量
min_len = min(len(imgs) for imgs in images_dict.values())
for prefix in images_dict:
    images_dict[prefix] = images_dict[prefix][:min_len]

print(f"[INFO] 图像对数量: {min_len}")
index = 0

def add_label(img, label):
    labeled_img = img.copy()
    return cv2.putText(
        labeled_img,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255 if img.ndim == 2 else (255, 255, 255),
        2
    )

def show_all(index):
    base_size = None
    labeled_images = []

    for prefix in folders.keys():  # 按照定义顺序显示
        img_path = images_dict[prefix][index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if base_size is None:
            base_size = (img.shape[1], img.shape[0])
        else:
            img = cv2.resize(img, base_size)

        labeled_img = add_label(img, prefix)
        labeled_images.append(labeled_img)

    # 拼接所有图像
    combined = np.hstack(labeled_images)
    cv2.imshow("Compare GT + Models", combined)

# ✅ 键盘控制
while True:
    show_all(index)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == ord('a') or key == 81:  # ←
        index = (index - 1) % min_len
    elif key == ord('d') or key == 83:  # →
        index = (index + 1) % min_len

cv2.destroyAllWindows()
