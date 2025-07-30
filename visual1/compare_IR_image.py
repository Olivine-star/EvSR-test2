# import os
# import cv2
# import numpy as np
# import glob

# # ✅ 所有路径和前缀
# folders = {
#     "GT": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\images",
#     "HR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HR\E2VID_Results",
#     "LR": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\LR\E2VID_Results",
#     "BS": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre\E2VID_Results",
#     "light": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light\E2VID_Results",
#     "light-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-light-p-learn\E2VID_Results",
#     "Louck-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre-Louck-light-p\E2VID_Results",
#     "Louck-l-l": r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000\boxes_6dof\SR_Test\HRPre2(Louck-light-p-learn)\E2VID_Results"
# }

# # ✅ 获取所有图片列表并排序
# images_dict = {}
# for prefix, path in folders.items():
#     img_list = sorted(glob.glob(os.path.join(path, "*.png")))
#     images_dict[prefix] = img_list

# # ✅ 统一对比图数量
# min_len = min(len(imgs) for imgs in images_dict.values())
# for prefix in images_dict:
#     images_dict[prefix] = images_dict[prefix][:min_len]

# print(f"[INFO] 图像对数量: {min_len}")
# index = 0

# def add_label(img, prefix, filename):
#     # 获取高度宽度
#     h, w = img.shape[:2]
    
#     # 创建上方和下方的文字条区域
#     bar_height = 30
#     top_bar = np.full((bar_height, w), 0, dtype=np.uint8)         # 上：prefix
#     bottom_bar = np.full((bar_height, w), 0, dtype=np.uint8)      # 下：filename

#     # 加上文字（白色）
#     cv2.putText(top_bar, prefix, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
#     cv2.putText(bottom_bar, filename, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)

#     # 拼接：上 + 图像 + 下
#     labeled_img = np.vstack((top_bar, img, bottom_bar))
#     return labeled_img
# def show_all(index):
#     base_size = None
#     labeled_images = []

#     for prefix in folders.keys():  # 保持顺序
#         img_path = images_dict[prefix][index]
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         filename = os.path.basename(img_path)

#         if base_size is None:
#             base_size = (img.shape[1], img.shape[0])
#         else:
#             img = cv2.resize(img, base_size)

#         labeled_img = add_label(img, prefix, filename)
#         labeled_images.append(labeled_img)

#     # 将图像按两行（每行4张）分组
#     row1 = labeled_images[:4]
#     row2 = labeled_images[4:]

#     # 横向拼接每一行
#     combined_row1 = np.hstack(row1)
#     combined_row2 = np.hstack(row2)

#     # 纵向拼接两行
#     final_image = np.vstack([combined_row1, combined_row2])
#     cv2.imshow("Compare GT + Models (2 Rows)", final_image)


# # ✅ 键盘控制
# while True:
#     show_all(index)
#     key = cv2.waitKey(0)

#     if key == 27:  # ESC
#         break
#     elif key == ord('a') or key == 81:  # ←
#         index = (index - 1) % min_len
#     elif key == ord('d') or key == 83:  # →
#         index = (index + 1) % min_len

# cv2.destroyAllWindows()




import os
import cv2
import numpy as np
import glob

# ✅ 输入：根目录 & 场景类别名
root_dir = r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test2-50000"
scene = "shapes_6dof"  # ← 修改为 dynamic_6dof、poster_6dof 等

# ✅ 构建前缀和路径映射
folders = {
    "GT": os.path.join(root_dir, scene, "images"),
    "HR": os.path.join(root_dir, scene, "SR_Test", "HR", "E2VID_Results"),
    "LR": os.path.join(root_dir, scene, "SR_Test", "LR", "E2VID_Results"),
    "BS": os.path.join(root_dir, scene, "SR_Test", "HRPre", "E2VID_Results"),
    "light": os.path.join(root_dir, scene, "SR_Test", "HRPre-light", "E2VID_Results"),
    "light-l": os.path.join(root_dir, scene, "SR_Test", "HRPre-light-p-learn", "E2VID_Results"),
    "Louck-l": os.path.join(root_dir, scene, "SR_Test", "HRPre-Louck-light-p", "E2VID_Results"),
    "Louck-l-l": os.path.join(root_dir, scene, "SR_Test", "HRPre2(Louck-light-p-learn)", "E2VID_Results")
}

# ✅ 读取并排序每类图像
images_dict = {}
for prefix, path in folders.items():
    img_list = sorted(glob.glob(os.path.join(path, "*.png")))
    if len(img_list) == 0:
        raise FileNotFoundError(f"[ERROR] 没有在路径中找到图片: {path}")
    images_dict[prefix] = img_list

# ✅ 保留相同最小长度
min_len = min(len(imgs) for imgs in images_dict.values())
for prefix in images_dict:
    images_dict[prefix] = images_dict[prefix][:min_len]

print(f"[INFO] 成功加载 {len(folders)} 个模型的图像，每类 {min_len} 张")

index = 0

def add_label(img, prefix, filename):
    h, w = img.shape[:2]
    bar_height = 30
    top_bar = np.full((bar_height, w), 0, dtype=np.uint8)
    bottom_bar = np.full((bar_height, w), 0, dtype=np.uint8)
    cv2.putText(top_bar, prefix, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(bottom_bar, filename, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)
    return np.vstack((top_bar, img, bottom_bar))

def show_all(index):
    base_size = None
    labeled_images = []

    for prefix in folders.keys():  # 保持顺序
        img_path = images_dict[prefix][index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.basename(img_path)

        if base_size is None:
            base_size = (img.shape[1], img.shape[0])
        else:
            img = cv2.resize(img, base_size)

        labeled_img = add_label(img, prefix, filename)
        labeled_images.append(labeled_img)

    # 将图像按两行（每行4张）分组
    row1 = labeled_images[:4]
    row2 = labeled_images[4:]

    # 横向拼接每一行
    combined_row1 = np.hstack(row1)
    combined_row2 = np.hstack(row2)

    # 纵向拼接两行
    final_image = np.vstack([combined_row1, combined_row2])
    cv2.imshow("Compare GT + Models (2 Rows)", final_image)


# ✅ 键盘控制
while True:
    show_all(index)
    key = cv2.waitKey(0)

    if key == 27:  # ESC 退出
        break
    elif key == ord('a') or key == 81:  # 左
        index = (index - 1) % min_len
    elif key == ord('d') or key == 83:  # 右
        index = (index + 1) % min_len

cv2.destroyAllWindows()
