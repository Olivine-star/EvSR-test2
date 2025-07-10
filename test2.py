import numpy as np
import os

def temporalSubsample(eventData, numSubstreams=5):
    """
    将事件数据按相等时间间隔分成多个子流
    
    Args:
        eventData: numpy数组，形状为(N, 4)，每行为[t, x, y, p]
        numSubstreams: 子流数量，默认为5
    
    Returns:
        list: 包含numSubstreams个子流的列表，每个子流都是numpy数组
    """
    if len(eventData) == 0:
        return [np.array([]).reshape(0, 4) for _ in range(numSubstreams)]
    
    # 获取时间范围
    minTime = eventData[:, 0].min()
    maxTime = eventData[:, 0].max()
    timeRange = maxTime - minTime
    
    print(f"时间范围: {minTime:.2f} - {maxTime:.2f} (总时长: {timeRange:.2f})")
    
    # 如果时间范围为0，所有事件都在同一时刻
    if timeRange == 0:
        # 将所有事件分配给第一个子流，其他子流为空
        substreams = [eventData.copy()]
        for _ in range(numSubstreams - 1):
            substreams.append(np.array([]).reshape(0, 4))
        return substreams
    
    # 计算每个子流的时间间隔
    intervalSize = timeRange / numSubstreams
    substreams = []
    
    for i in range(numSubstreams):
        # 计算当前子流的时间边界
        startTime = minTime + i * intervalSize
        endTime = minTime + (i + 1) * intervalSize
        
        # 对于最后一个子流，包含最大时间点
        if i == numSubstreams - 1:
            mask = (eventData[:, 0] >= startTime) & (eventData[:, 0] <= endTime)
        else:
            mask = (eventData[:, 0] >= startTime) & (eventData[:, 0] < endTime)
        
        # 提取当前子流的事件
        substream = eventData[mask].copy()
        substreams.append(substream)
        
        print(f"子流 {i+1}: 时间范围 [{startTime:.2f}, {endTime:.2f}], 事件数量: {len(substream)}")
    
    return substreams

def analyzeEventData(eventData, title="事件数据"):
    """
    分析事件数据的基本信息
    
    Args:
        eventData: numpy数组，形状为(N, 4)，每行为[t, x, y, p]
        title: 数据标题
    """
    print(f"\n=== {title} 分析 ===")
    print(f"数据形状: {eventData.shape}")
    print(f"数据类型: {eventData.dtype}")
    
    if len(eventData) > 0:
        print(f"时间范围: {eventData[:, 0].min():.2f} - {eventData[:, 0].max():.2f}")
        print(f"X坐标范围: {eventData[:, 1].min():.0f} - {eventData[:, 1].max():.0f}")
        print(f"Y坐标范围: {eventData[:, 2].min():.0f} - {eventData[:, 2].max():.0f}")
        print(f"极性分布: {np.bincount(eventData[:, 3].astype(int))}")
        
        print(f"\n前5个事件:")
        for i in range(min(5, len(eventData))):
            t, x, y, p = eventData[i]
            print(f"  事件 {i+1}: [t={t:8.2f}, x={x:3.0f}, y={y:3.0f}, p={p:1.0f}]")
        
        if len(eventData) > 5:
            print(f"\n后5个事件:")
            for i in range(max(0, len(eventData)-5), len(eventData)):
                t, x, y, p = eventData[i]
                print(f"  事件 {i+1}: [t={t:8.2f}, x={x:3.0f}, y={y:3.0f}, p={p:1.0f}]")
    else:
        print("数据为空")

def loadEventData(filepath):
    """
    加载事件数据文件
    
    Args:
        filepath: .npy文件路径
    
    Returns:
        numpy数组: 事件数据
    """
    try:
        data = np.load(filepath)
        print(f"成功加载文件: {filepath}")
        return data
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

def main():
    """
    主函数：测试事件数据的时间分割功能
    """
    # 事件数据文件路径
    filepath = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR\0\0.npy"
    
    print("=" * 80)
    print("事件数据时间分割测试")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在 - {filepath}")
        print("请检查文件路径是否正确")
        return
    
    # 加载事件数据
    eventData = loadEventData(filepath)
    if eventData is None:
        return
    
    # 分析原始数据
    analyzeEventData(eventData, "原始事件数据")
    
    # 进行时间分割
    print("\n" + "=" * 60)
    print("开始时间分割...")
    print("=" * 60)
    
    substreams = temporalSubsample(eventData, numSubstreams=5)
    
    # 分析分割后的数据
    print("\n" + "=" * 60)
    print("分割后的子流分析")
    print("=" * 60)
    
    totalEvents = 0
    for i, substream in enumerate(substreams):
        analyzeEventData(substream, f"子流 {i+1}")
        totalEvents += len(substream)
        print("-" * 40)
    
    # 验证数据完整性
    print(f"\n=== 数据完整性验证 ===")
    print(f"原始事件数量: {len(eventData)}")
    print(f"分割后总事件数量: {totalEvents}")
    print(f"数据完整性: {'✓ 通过' if totalEvents == len(eventData) else '✗ 失败'}")
    
    # 验证时间连续性
    print(f"\n=== 时间连续性验证 ===")
    if len(eventData) > 0:
        originalTimeRange = eventData[:, 0].max() - eventData[:, 0].min()
        print(f"原始时间范围: {originalTimeRange:.2f}")
        
        # 检查子流时间覆盖
        for i, substream in enumerate(substreams):
            if len(substream) > 0:
                subTimeMin = substream[:, 0].min()
                subTimeMax = substream[:, 0].max()
                print(f"子流 {i+1} 时间范围: {subTimeMin:.2f} - {subTimeMax:.2f}")

if __name__ == "__main__":
    main()
