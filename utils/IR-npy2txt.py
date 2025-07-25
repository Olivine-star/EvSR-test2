# import os
# import numpy as np

# # 设置你的文件夹路径
# npy_dir = r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test\dynamic_6dof\SR_Test\HRPre"
# output_txt_path = os.path.join(npy_dir, "IR.txt")

# # 获取目录下所有 .npy 文件，并按文件名数字排序
# npy_files = sorted(
#     [f for f in os.listdir(npy_dir) if f.endswith(".npy")],
#     key=lambda x: int(os.path.splitext(x)[0])  # 提取数字部分用于排序
# )

# # 打开输出文件
# with open(output_txt_path, "w") as out_file:
#     for npy_file in npy_files:
#         npy_path = os.path.join(npy_dir, npy_file)
#         try:
#             events = np.load(npy_path)  # shape: [N, 4]
#             for event in events:
#                 line = " ".join(map(str, event)) + "\n"
#                 out_file.write(line)
#         except Exception as e:
#             print(f"[ERROR] Failed to load {npy_path}: {e}")

# print(f"[INFO] 合并完成，共写入 {output_txt_path}")




import os
import numpy as np

npy_dir = r"C:\code\EventSR-Project\EvSR-test2\dataset\ImageReconstruction\test\dynamic_6dof\SR_Test\HRPre"
output_txt_path = os.path.join(npy_dir, "IR.txt")

# ✅ 设置图像分辨率（必须）
sensor_width = 346
sensor_height = 260

npy_files = sorted(
    [f for f in os.listdir(npy_dir) if f.endswith(".npy")],
    key=lambda x: int(os.path.splitext(x)[0])
)

with open(output_txt_path, "w") as out_file:
    # ✅ 写入第一行尺寸
    out_file.write(f"{sensor_width} {sensor_height}\n")

    for npy_file in npy_files:
        npy_path = os.path.join(npy_dir, npy_file)
        try:
            events = np.load(npy_path)  # shape: [N, 4]
            for event in events:
                line = " ".join(map(str, event)) + "\n"
                out_file.write(line)
        except Exception as e:
            print(f"[ERROR] Failed to load {npy_path}: {e}")

print(f"[INFO] 合并完成，共写入 {output_txt_path}")
