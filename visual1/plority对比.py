import numpy as np

# 加载事件数据
"""
该数据集时间戳是0-300多ms
"""

path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\ResConv\HRPre\0\0.npy"
path2 = r'D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR\0\0.npy'
events = np.load(path)
events2 = np.load(path2)

# 检查加载后的数据结构-
print("数据总数:", events.shape[0])
print("\n前五个事件：")
print(events[:5])

print("\n后五个事件：")
print(events[-5:])

print("================ground truth==================")

print("\n数据总数:", events2.shape[0])
print("\n前五个事件：")
print(events2[:5])

print("\n后五个事件：")
print(events2[-5:])