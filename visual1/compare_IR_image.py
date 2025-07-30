import os
import cv2
import numpy as np
import glob

# ✅ 定义路径和前缀名称
folders = {
    "HR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HR\E2VID_Results",
    "LR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\LR\E2VID_Results",
    "BS": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre\E2VID_Results",
    "light": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light\E2VID_Results",
    "light-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light-p-learn\E2VID_Results",
    "Louck-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-Louck-light-p\E2VID_Results",
    "Louck-l-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre2(Louck-light-p-learn)\E2VID_Results"
}

# ✅ 读取每个文件夹中的图片路径并排序
images_dict = {}
for prefix, path in folders.items():
    img_list = sorted(glob.glob(os.path.join(path, "*.png")))
    images_dict[prefix] = img_list

# ✅ 取最小长度以避免越界
min_len = min(len(imgs) for imgs in images_dict.values())

print(f"[INFO] 有效对比图像数量: {min_len}")
index = 0

def add_label(img, label):
    # 在图像上方添加标签文字
    labeled_img = img.copy()
    labeled_img = cv2.putText(
        labeled_img,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255 if img.ndim == 2 else (255, 255, 255),
        2
    )
    return labeled_img

def show_all(index):
    resized_images = []
    base_size = None

    for prefix, img_list in images_dict.items():
        img = cv2.imread(img_list[index], cv2.IMREAD_GRAYSCALE)
        if base_size is None:
            base_size = (img.shape[1], img.shape[0])  # (width, height)
        else:
            img = cv2.resize(img, base_size)

        labeled_img = add_label(img, prefix)
        resized_images.append(labeled_img)

    # 横向拼接
    combined = np.hstack(resized_images)
    cv2.imshow("Compare All Versions", combined)

# ✅ 控制逻辑：a/d 切换，ESC 退出
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
