import torch
import slayerSNN as snn
from utils.utils import getNeuronConfig
import numpy as np


import os
from slayerSNN.spikeFileIO import event

def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

#
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Step 1: Load event camera data from .npy file
# Event data format: [timestamp, x, y, polarity]
print("=== Step 1: Loading Event Camera Data ===")
event_file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\LR\0\0.npy"
print(f"Loading events from: {event_file_path}")

# Load raw numpy array first to examine the data structure
raw_events = np.load(event_file_path)
print(f"Raw event data shape: {raw_events.shape}")
print(f"Raw event data type: {raw_events.dtype}")
print(f"First 5 events:\n{raw_events[:5]}")
print(f"Event ranges - X: [{raw_events[:, 1].min()}, {raw_events[:, 1].max()}], "
      f"Y: [{raw_events[:, 2].min()}, {raw_events[:, 2].max()}], "
      f"Time: [{raw_events[:, 0].min()}, {raw_events[:, 0].max()}], "
      f"Polarity: [{raw_events[:, 3].min()}, {raw_events[:, 3].max()}]")

# Convert to slayerSNN event object
events = readNpSpikes(event_file_path)
print(f"Converted to slayerSNN event object")
print(f"Event object attributes: x, y, t, p")

print("\n=== Step 2: Creating Empty Tensor Container ===")
# Step 2: Create appropriately sized empty tensor as container
# Tensor dimensions: [Channels, Height, Width, Time]
# - Channels: 2 (positive and negative polarity)
# - Height: 17 pixels (based on low-resolution N-MNIST)
# - Width: 17 pixels
# - Time: 350 time steps
C, H, W, T = 2, 17, 17, 350
emptyTensor = torch.zeros((C, H, W, T))
print(f"Empty tensor shape: {emptyTensor.shape}")
print(f"Empty tensor dimensions: [Channels={C}, Height={H}, Width={W}, Time={T}]")
print(f"Empty tensor data type: {emptyTensor.dtype}")
print(f"Total tensor elements: {emptyTensor.numel()}")
print(f"Memory usage: {emptyTensor.numel() * 4 / 1024 / 1024:.2f} MB (float32)")

print("\n=== Step 3: Converting Sparse Events to Dense Spike Tensor ===")
# Step 3: Convert sparse event data into dense 4D tensor format
# The toSpikeTensor method fills the empty tensor with spike data:
# - Events with positive polarity go to channel 0
# - Events with negative polarity go to channel 1
# - Each event creates a spike at position [channel, y, x, t]
spike_tensor = events.toSpikeTensor(emptyTensor)
print(f"Spike tensor shape after conversion: {spike_tensor.shape}")
print(f"Spike tensor data type: {spike_tensor.dtype}")

print("\n=== Step 4: Adding Batch Dimension and Moving to GPU ===")
# Step 4: Add batch dimension for neural network processing
# Most neural networks expect input format: [Batch, Channels, Height, Width, Time]
spike_tensor_batched = torch.unsqueeze(spike_tensor, dim=0).cuda()
print(f"Final tensor shape with batch dimension: {spike_tensor_batched.shape}")
print(f"Final tensor device: {spike_tensor_batched.device}")
print(f"Final tensor data type: {spike_tensor_batched.dtype}")

print("\n=== Step 5: Analyzing Tensor Sparsity and Content ===")
# Step 5: Analyze the resulting tensor to understand sparsity
total_elements = spike_tensor_batched.numel()
non_zero_elements = torch.count_nonzero(spike_tensor_batched).item()
sparsity_ratio = (total_elements - non_zero_elements) / total_elements * 100

print(f"Total tensor elements: {total_elements}")
print(f"Non-zero elements: {non_zero_elements}")
print(f"Zero elements: {total_elements - non_zero_elements}")
print(f"Sparsity ratio: {sparsity_ratio:.2f}% (percentage of zeros)")
print(f"Data density: {100 - sparsity_ratio:.2f}% (percentage of non-zeros)")

# Analyze distribution across channels
channel_0_spikes = torch.count_nonzero(spike_tensor_batched[0, 0]).item()
channel_1_spikes = torch.count_nonzero(spike_tensor_batched[0, 1]).item()
print(f"Spikes in channel 0 (positive polarity): {channel_0_spikes}")
print(f"Spikes in channel 1 (negative polarity): {channel_1_spikes}")

print("\n=== Step 6: Sample Tensor Values ===")
# Show a small sample of the tensor to visualize the data structure
print("Sample from channel 0 (positive polarity), first time step:")
print(spike_tensor_batched[0, 0, :5, :5, 0])
print("\nSample from channel 1 (negative polarity), first time step:")
print(spike_tensor_batched[0, 1, :5, :5, 0])

# Find and display some non-zero values
if non_zero_elements > 0:
    print(f"\nFirst few non-zero spike locations and values:")
    nonzero_indices = torch.nonzero(spike_tensor_batched)[:10]  # Get first 10 non-zero locations
    for i, idx in enumerate(nonzero_indices):
        batch, channel, y, x, t = idx
        value = spike_tensor_batched[batch, channel, y, x, t]
        polarity = "positive" if channel == 0 else "negative"
        print(f"  Spike {i+1}: [{polarity}] at position (y={y}, x={x}, t={t}) = {value:.3f}")

print(f"\n=== Conversion Complete ===")
print(f"Successfully converted {len(raw_events)} sparse events into dense 4D tensor")
print(f"Tensor ready for spiking neural network processing!")





print("=== Step 6: 非零脉冲位置张量 ===")
nonzero_indices = torch.nonzero(spike_tensor_batched)  # 所有非零位置
nonzero_values = spike_tensor_batched[nonzero_indices[:, 0],
                                      nonzero_indices[:, 1],
                                      nonzero_indices[:, 2],
                                      nonzero_indices[:, 3],
                                      nonzero_indices[:, 4]]

# 拼接索引和数值，得到 [N, 6] 张量：batch, channel, y, x, t, value
nonzero_info = torch.cat([nonzero_indices, nonzero_values.unsqueeze(1)], dim=1)
# tensor([[  0.,   0.,   2.,   6., 281.,   1.], 代表第0个batch，第0通道（极性为positive polarity），y=2，x=6，时间戳是281,1表示发生一个脉冲
print("非零 spike 张量索引和值（前 66 行）：")

print(nonzero_info[:66])
