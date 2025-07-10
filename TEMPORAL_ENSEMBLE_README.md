# Temporal Ensemble Processing for N-MNIST Super-Resolution

This implementation extends the original N-MNIST super-resolution framework with temporal subsampling and multi-inference processing capabilities. The enhanced pipeline divides event streams into temporal sub-streams and processes them through ensemble inference for improved performance.

## Overview

### Key Features

1. **Temporal Subsampling**: Divides input event streams into 5 equal temporal sub-streams based on timestamp ranges
2. **Multi-Inference Processing**: Performs separate forward passes through the model for each sub-stream
3. **Output Aggregation**: Combines results using configurable aggregation methods (sum, mean, max, weighted mean)
4. **Enhanced Training**: Temporal ensemble training with improved loss computation
5. **Comprehensive Evaluation**: Detailed quality metrics and performance analysis

### Architecture

```
Input Event Stream
       ↓
Temporal Subsampling (5 sub-streams)
       ↓
┌─────────────────────────────────────┐
│ Sub-stream 1 → Model → Output 1     │
│ Sub-stream 2 → Model → Output 2     │
│ Sub-stream 3 → Model → Output 3     │
│ Sub-stream 4 → Model → Output 4     │
│ Sub-stream 5 → Model → Output 5     │
└─────────────────────────────────────┘
       ↓
Output Aggregation (sum/mean/max)
       ↓
Final Super-Resolution Result
```

## File Structure

### New Files (with "_1" suffix)

- **`nMnist/mnistDatasetSR_1.py`**: Enhanced dataset loader with temporal subsampling
- **`nMnist/trainNmnist_1.py`**: Training script with multi-inference processing
- **`nMnist/testNmnist_1.py`**: Testing script with temporal ensemble evaluation
- **`model_1.py`**: Enhanced model architectures with temporal ensemble support

### Key Components

#### 1. Temporal Subsampling (`mnistDatasetSR_1.py`)

```python
def temporalSubsample(eventObj, numSubstreams=5):
    """
    Divide an event stream into equal temporal sub-streams.
    
    - Preserves chronological order of events
    - Creates 5 equal time windows
    - Handles edge cases (empty sub-streams)
    """
```

**Features:**
- Automatic time range calculation
- Equal temporal division
- Chronological order preservation
- Empty sub-stream handling

#### 2. Multi-Inference Training (`trainNmnist_1.py`)

```python
class TemporalEnsembleTrainer:
    """
    Enhanced trainer with temporal ensemble processing.
    
    - Processes 5 sub-streams per sample
    - Configurable aggregation methods
    - Enhanced loss computation
    """
```

**Training Process:**
1. Load temporal sub-streams from dataset
2. Process each sub-stream through model
3. Aggregate outputs using specified method
4. Compute loss across all sub-streams
5. Backpropagate and update parameters

#### 3. Enhanced Models (`model_1.py`)

```python
class NetworkBasicTemporal(torch.nn.Module):
    """
    Temporal-aware NetworkBasic with built-in ensemble support.
    
    - Native temporal processing
    - Multiple aggregation strategies
    - Backward compatibility
    """
```

**Aggregation Methods:**
- **Sum**: Element-wise summation of outputs
- **Mean**: Element-wise averaging of outputs  
- **Max**: Element-wise maximum of outputs
- **Weighted Mean**: Learnable weighted averaging

## Usage

### Training with Temporal Ensemble

```python
# Configuration
config = {
    'num_substreams': 5,
    'aggregation_method': 'mean',  # 'sum', 'mean', 'max', 'weighted_mean'
    'batch_size': 8,
    'max_epochs': 30,
    'model_class': 'NetworkBasicTemporal'
}

# Initialize and train
trainer = TemporalEnsembleTrainer(config)
trainer.train()
```

### Testing with Temporal Ensemble

```python
# Configuration
config = {
    'num_substreams': 5,
    'aggregation_method': 'mean',
    'model_class': 'NetworkBasicTemporal',
    'checkpoint_path': './ckpt_temporal_ensemble'
}

# Initialize and test
tester = TemporalEnsembleTester(config)
tester.run_evaluation()
```

### Dataset Usage

```python
# Load dataset with temporal subsampling
dataset = mnistDatasetTemporal(
    train=True,
    numSubstreams=5
)

# Get sample with 5 temporal sub-streams
lr_substreams, hr_substreams = dataset[0]
# lr_substreams: List of 5 LR tensors [2, 17, 17, sub_time_bins]
# hr_substreams: List of 5 HR tensors [2, 34, 34, sub_time_bins]
```

## Configuration Options

### Temporal Parameters

- **`num_substreams`**: Number of temporal sub-streams (default: 5)
- **`aggregation_method`**: Output aggregation strategy
  - `'sum'`: Element-wise summation
  - `'mean'`: Element-wise averaging (recommended)
  - `'max'`: Element-wise maximum
  - `'weighted_mean'`: Learnable weighted averaging

### Model Options

- **`NetworkBasicTemporal`**: Enhanced basic architecture
- **`Network1Temporal`**: Enhanced Network1 with temporal support
- **`TemporalEnsembleWrapper`**: Wrapper for existing models

### Training Parameters

- **`batch_size`**: Batch size (recommended: 8 for temporal ensemble)
- **`learning_rate`**: Learning rate (default: 0.001)
- **`max_epochs`**: Maximum training epochs (default: 30)

## Performance Benefits

### Expected Improvements

1. **Temporal Consistency**: Better temporal coherence in outputs
2. **Noise Reduction**: Ensemble averaging reduces noise
3. **Robustness**: More stable performance across different samples
4. **Quality Enhancement**: Improved super-resolution quality

### Computational Considerations

- **Memory Usage**: ~5x increase due to multiple sub-streams
- **Training Time**: ~5x increase due to multiple forward passes
- **Inference Time**: ~5x increase but with quality improvements

## Evaluation Metrics

The testing script provides comprehensive evaluation:

```python
metrics = {
    'mse': 0.001234,                    # Mean Squared Error
    'spike_ratio': 0.95,                # Output/Target spike ratio
    'output_spikes': 1250.0,            # Average output spikes
    'target_spikes': 1315.0,            # Average target spikes
    'output_temporal_var': 0.0045,      # Temporal variance
    'target_temporal_var': 0.0052       # Target temporal variance
}
```

## Best Practices

### Recommended Settings

1. **For Training**:
   - Use `aggregation_method='mean'` for stable training
   - Start with `batch_size=8` and adjust based on GPU memory
   - Use `num_substreams=5` for good temporal resolution

2. **For Testing**:
   - Use same aggregation method as training
   - Process samples individually (`batch_size=1`)
   - Save detailed metrics for analysis

3. **For Model Selection**:
   - Start with `NetworkBasicTemporal` for baseline
   - Try `Network1Temporal` for more complex scenarios
   - Use `TemporalEnsembleWrapper` for existing models

### Troubleshooting

1. **Memory Issues**: Reduce batch size or number of sub-streams
2. **Training Instability**: Use `aggregation_method='mean'` instead of 'sum'
3. **Slow Training**: Consider reducing `num_substreams` to 3
4. **Poor Quality**: Ensure temporal division preserves important events

## Backward Compatibility

All new files maintain backward compatibility:
- Original model classes are aliased to temporal versions
- Configuration files remain unchanged
- Existing checkpoints can be loaded (with some limitations)

## Future Enhancements

Potential improvements:
1. **Adaptive Temporal Division**: Non-uniform time windows based on event density
2. **Attention Mechanisms**: Learnable attention weights for sub-streams
3. **Hierarchical Processing**: Multi-scale temporal processing
4. **Dynamic Sub-streams**: Variable number of sub-streams per sample

## Citation

If you use this temporal ensemble implementation, please cite the original EventSR work and mention the temporal enhancement:

```
Enhanced with temporal subsampling and multi-inference processing
for improved event camera super-resolution performance.
```
