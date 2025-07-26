# import os
# import numpy as np

# # ✅ 根目录
# root_dir = r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test\boxes_6dof\SR_Test"

# # ✅ 图像尺寸
# sensor_width = 346
# sensor_height = 260

# # ✅ 遍历 SR_Test 下所有子目录
# for subfolder in os.listdir(root_dir):
#     subfolder_path = os.path.join(root_dir, subfolder)
#     if not os.path.isdir(subfolder_path):
#         continue

#     npy_files = sorted(
#         [f for f in os.listdir(subfolder_path) if f.endswith(".npy")],
#         key=lambda x: int(os.path.splitext(x)[0])
#     )

#     output_txt_path = os.path.join(subfolder_path, "IR.txt")
#     with open(output_txt_path, "w") as out_file:
#         # 写入尺寸
#         out_file.write(f"{sensor_width} {sensor_height}\n")

#         print(f"\n[INFO]  正在处理文件夹: {subfolder} （共 {len(npy_files)} 个 .npy 文件）")

#         for idx, npy_file in enumerate(npy_files):
#             npy_path = os.path.join(subfolder_path, npy_file)
#             try:
#                 events = np.load(npy_path)  # shape: [N, 4]
#                 for event in events:
#                     out_file.write(" ".join(map(str, event)) + "\n")
#                 print(f"  [{idx+1:4d}/{len(npy_files)}] 已处理 {npy_file}，事件数: {len(events)}")
#             except Exception as e:
#                 print(f"  [ERROR] 无法加载 {npy_file}：{e}")

#     print(f"[ DONE] 合并完成: {output_txt_path}")




import os
import numpy as np
import torch

# ✅ 根目录
root_dir = r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test\calibration\SR_Test"

# ✅ 图像尺寸
sensor_width = 346
sensor_height = 260

# ✅ 是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

# ✅ 遍历 SR_Test 下所有子目录
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    npy_files = sorted(
        [f for f in os.listdir(subfolder_path) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    output_txt_path = os.path.join(subfolder_path, "IR.txt")
    with open(output_txt_path, "w") as out_file:
        # 写入图像尺寸
        out_file.write(f"{sensor_width} {sensor_height}\n")

        print(f"\n[INFO] 正在处理文件夹: {subfolder} （共 {len(npy_files)} 个 .npy 文件）")

        for idx, npy_file in enumerate(npy_files):
            npy_path = os.path.join(subfolder_path, npy_file)
            try:
                # ➤ 使用 torch 加载数据并放入 GPU
                events_np = np.load(npy_path)  # [N, 4], still load via numpy
                events_tensor = torch.tensor(events_np, dtype=torch.int32, device=device)  # Move to GPU

                # ➤ 将事件 tensor 移回 CPU 再写入文本（避免 I/O 与 GPU 同步瓶颈）
                lines = events_tensor.cpu().numpy()
                lines_str = "\n".join([" ".join(map(str, row)) for row in lines]) + "\n"
                out_file.write(lines_str)

                print(f"  [{idx+1:4d}/{len(npy_files)}] 已处理 {npy_file}，事件数: {events_tensor.shape[0]}")
            except Exception as e:
                print(f"  [ERROR] 无法加载 {npy_file}：{e}")

    print(f"[ DONE] 合并完成: {output_txt_path}")
