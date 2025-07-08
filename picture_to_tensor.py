from PIL import Image
import torchvision.transforms as transforms
import torch

# 步骤1：设置图片路径
image_path = r"C:\Users\zhx\Desktop\1.jpg"  # 替换为你本地的图片路径

# 步骤2：读取图片
image = Image.open(image_path).convert('RGB')  # 转换为RGB格式

# 步骤3：定义转换为张量的transform
transform = transforms.ToTensor()

# 步骤4：应用transform，将图片转换为张量
tensor_image = transform(image)

# 步骤5：打印张量
print("图片转为张量后的结果：")
print(tensor_image)
print("张量的形状：", tensor_image.shape)  # 一般为 [C, H, W]




# 步骤5：分别打印 R, G, B 通道
print("图片转为张量后的结果：")
print("张量的整体形状：", tensor_image.shape)  # 一般为 [C, H, W]

print("\n=== R通道（Red） ===")
print(tensor_image[0])
print("R通道形状：", tensor_image[0].shape)

print("\n=== G通道（Green） ===")
print(tensor_image[1])
print("G通道形状：", tensor_image[1].shape)

print("\n=== B通道（Blue） ===")
print(tensor_image[2])
print("B通道形状：", tensor_image[2].shape)