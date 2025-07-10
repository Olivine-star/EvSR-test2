"""
Enhanced N-MNIST testing script with temporal subsampling and multi-inference processing.

This script extends the original testNmnist.py to support:
1. Temporal subsampling for improved inference quality
2. Multi-inference processing across temporal sub-streams
3. Configurable output aggregation methods
4. Comprehensive evaluation metrics

Key Features:
- Processes test data with temporal ensemble inference
- Saves aggregated super-resolution results
- Provides detailed performance analysis
- Supports multiple aggregation strategies
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import os
import datetime
from torch.utils.data import DataLoader
from mnistDatasetSR_1 import mnistDatasetTemporal
from model_1 import NetworkBasicTemporal, Network1Temporal
from utils.ckpt import checkpoint_restore
from utils.utils import getEventFromTensor
from slayerSNN.spikeFileIO import event
import slayerSNN as snn

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'


class TemporalEnsembleTester:
    """
    Enhanced tester class for temporal subsampling and multi-inference evaluation.
    """
    
    def __init__(self, config):
        """
        Initialize the temporal ensemble tester.
        
        Args:
            config (dict): Configuration dictionary containing test parameters
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        self.numSubstreams = config.get('num_substreams', 5)
        self.aggregation_method = config.get('aggregation_method', 'mean')
        
        # Read dataset paths
        with open('dataset_path.txt', 'r') as f:
            lines = f.read().splitlines()
            self.paths = {line.split('=')[0].strip(): line.split('=')[1].strip() 
                         for line in lines if '=' in line}
        
        # Initialize test dataset
        self.testDataset = mnistDatasetTemporal(
            train=False,
            numSubstreams=self.numSubstreams
        )
        
        # Initialize data loader
        self.batch_size = config.get('batch_size', 1)  # Use batch_size=1 for testing
        self.testLoader = DataLoader(
            dataset=self.testDataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # Initialize model
        netParams = snn.params(config.get('network_config', 'network.yaml'))
        model_class = config.get('model_class', 'NetworkBasicTemporal')
        
        if model_class == 'NetworkBasicTemporal':
            self.model = NetworkBasicTemporal(
                netParams, 
                num_substreams=self.numSubstreams,
                aggregation_method=self.aggregation_method
            )
        elif model_class == 'Network1Temporal':
            self.model = Network1Temporal(
                netParams,
                num_substreams=self.numSubstreams,
                aggregation_method=self.aggregation_method
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        ckpt_path = config.get('checkpoint_path', './ckpt_temporal_ensemble')
        self.model, _ = checkpoint_restore(self.model, ckpt_path)
        self.model.eval()
        
        # Setup save path
        self.save_path = config.get('save_path', 
                                   self.paths.get('savepath', './results_temporal'))
        os.makedirs(self.save_path, exist_ok=True)
        
        print(f"Temporal Ensemble Tester initialized:")
        print(f"  - Number of sub-streams: {self.numSubstreams}")
        print(f"  - Aggregation method: {self.aggregation_method}")
        print(f"  - Model: {model_class}")
        print(f"  - Test samples: {len(self.testDataset)}")
        print(f"  - Save path: {self.save_path}")
    
    def process_sample(self, lr_substreams, sample_idx):
        """
        Process a single sample with temporal ensemble inference.
        
        Args:
            lr_substreams (list): List of LR spike tensors for each sub-stream
            sample_idx (int): Sample index for saving
        
        Returns:
            torch.Tensor: Aggregated super-resolution output
        """
        with torch.no_grad():
            # Move sub-streams to device
            lr_substreams_gpu = [tensor.to(self.device) for tensor in lr_substreams]
            
            # Multi-inference forward pass
            if self.model.num_substreams > 1:
                # Process as temporal ensemble
                output = self.model(lr_substreams_gpu)
            else:
                # Process single stream (concatenate all sub-streams)
                concatenated_input = torch.cat(lr_substreams_gpu, dim=-1)
                output = self.model(concatenated_input)
            
            return output
    
    def save_result(self, output_tensor, sample_path):
        """
        Save the super-resolution result as .npy file.
        
        Args:
            output_tensor (torch.Tensor): Model output tensor
            sample_path (str): Path to save the result
        """
        # Convert tensor to events
        events = getEventFromTensor(output_tensor.cpu())
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        
        # Save as .npy file
        np.save(sample_path, events)
    
    def evaluate_quality_metrics(self, output, target):
        """
        Evaluate quality metrics between output and target.
        
        Args:
            output (torch.Tensor): Model output
            target (torch.Tensor): Ground truth target
        
        Returns:
            dict: Dictionary containing quality metrics
        """
        with torch.no_grad():
            # Move to CPU for computation
            output_cpu = output.cpu()
            target_cpu = target.cpu()
            
            # Mean Squared Error
            mse = torch.nn.functional.mse_loss(output_cpu, target_cpu).item()
            
            # Spike count comparison
            output_spikes = output_cpu.sum().item()
            target_spikes = target_cpu.sum().item()
            spike_ratio = output_spikes / (target_spikes + 1e-8)
            
            # Temporal consistency (variance across time)
            output_temporal_var = output_cpu.var(dim=-1).mean().item()
            target_temporal_var = target_cpu.var(dim=-1).mean().item()
            
            metrics = {
                'mse': mse,
                'output_spikes': output_spikes,
                'target_spikes': target_spikes,
                'spike_ratio': spike_ratio,
                'output_temporal_var': output_temporal_var,
                'target_temporal_var': target_temporal_var
            }
            
            return metrics
    
    def run_evaluation(self):
        """
        Run complete evaluation with temporal ensemble processing.
        """
        print(f"\n=== Starting Temporal Ensemble Evaluation ===")
        print(f"Processing {len(self.testDataset)} test samples...")
        
        total_metrics = {
            'mse': 0.0,
            'output_spikes': 0.0,
            'target_spikes': 0.0,
            'spike_ratio': 0.0,
            'output_temporal_var': 0.0,
            'target_temporal_var': 0.0
        }
        
        start_time = datetime.datetime.now()
        
        for i, (lr_substreams_batch, hr_substreams_batch) in enumerate(self.testLoader):
            # Process single sample (batch_size=1)
            lr_substreams = [tensor[0] for tensor in lr_substreams_batch]
            hr_substreams = [tensor[0] for tensor in hr_substreams_batch]
            
            # Temporal ensemble inference
            output = self.process_sample(lr_substreams, i)
            
            # Aggregate ground truth for comparison
            hr_aggregated = torch.stack(hr_substreams).mean(dim=0).to(self.device)
            
            # Evaluate quality metrics
            metrics = self.evaluate_quality_metrics(output, hr_aggregated)
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Save result
            # Determine save path based on original dataset structure
            class_idx = i // (len(self.testDataset) // 10)  # Assuming 10 classes
            sample_idx = i % (len(self.testDataset) // 10)
            save_dir = os.path.join(self.save_path, str(class_idx))
            save_file = os.path.join(save_dir, f"{sample_idx}.npy")
            
            self.save_result(output, save_file)
            
            # Progress reporting
            if (i + 1) % 100 == 0 or i == len(self.testDataset) - 1:
                elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                avg_time_per_sample = elapsed_time / (i + 1)
                remaining_samples = len(self.testDataset) - (i + 1)
                eta = remaining_samples * avg_time_per_sample
                
                print(f"Processed {i+1}/{len(self.testDataset)} samples "
                      f"({(i+1)/len(self.testDataset)*100:.1f}%) - "
                      f"ETA: {eta/60:.1f} minutes")
                
                # Print current metrics
                current_mse = total_metrics['mse'] / (i + 1)
                current_spike_ratio = total_metrics['spike_ratio'] / (i + 1)
                print(f"  Current avg MSE: {current_mse:.6f}")
                print(f"  Current avg spike ratio: {current_spike_ratio:.3f}")
        
        # Calculate final average metrics
        num_samples = len(self.testDataset)
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        
        # Print final results
        print(f"\n=== Evaluation Results ===")
        print(f"Aggregation method: {self.aggregation_method}")
        print(f"Number of sub-streams: {self.numSubstreams}")
        print(f"Total samples processed: {num_samples}")
        print(f"Average MSE: {avg_metrics['mse']:.6f}")
        print(f"Average spike ratio (output/target): {avg_metrics['spike_ratio']:.3f}")
        print(f"Average output spikes per sample: {avg_metrics['output_spikes']:.1f}")
        print(f"Average target spikes per sample: {avg_metrics['target_spikes']:.1f}")
        print(f"Average output temporal variance: {avg_metrics['output_temporal_var']:.6f}")
        print(f"Average target temporal variance: {avg_metrics['target_temporal_var']:.6f}")
        
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Total evaluation time: {total_time/60:.1f} minutes")
        print(f"Average time per sample: {total_time/num_samples:.2f} seconds")
        
        # Save metrics to file
        metrics_file = os.path.join(self.save_path, 'evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Temporal Ensemble Evaluation Results\n")
            f.write(f"=====================================\n")
            f.write(f"Aggregation method: {self.aggregation_method}\n")
            f.write(f"Number of sub-streams: {self.numSubstreams}\n")
            f.write(f"Total samples: {num_samples}\n")
            f.write(f"Average MSE: {avg_metrics['mse']:.6f}\n")
            f.write(f"Average spike ratio: {avg_metrics['spike_ratio']:.3f}\n")
            f.write(f"Average output spikes: {avg_metrics['output_spikes']:.1f}\n")
            f.write(f"Average target spikes: {avg_metrics['target_spikes']:.1f}\n")
            f.write(f"Average output temporal variance: {avg_metrics['output_temporal_var']:.6f}\n")
            f.write(f"Average target temporal variance: {avg_metrics['target_temporal_var']:.6f}\n")
            f.write(f"Total evaluation time: {total_time/60:.1f} minutes\n")
        
        print(f"Results saved to: {self.save_path}")
        print(f"Metrics saved to: {metrics_file}")


def main():
    # 文件路径
    file_path = r'D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Test\HR\0\0.npy'

    # 加载原始数据：格式应为 [t, x, y, p]
    raw = np.load(file_path)
    print(f"✅ 原始事件数据 shape: {raw.shape}（每行表示一个事件）")
    print("第一条事件：[t, x, y, p] =", raw[0])

    # 构造事件对象（注意：slayerSNN.event 使用的顺序是 x, y, p, t）
    ev = event(
    raw[:, 1],              # x
    raw[:, 2],              # y
    raw[:, 3],              # polarity
    raw[:, 0] * 1e-3        # time in milliseconds
    )


    # 分割为 5 个时间子流
    substreams = temporalSubsample(ev, numSubstreams=5)

    # 打印每个子流的基本信息
    for i, sub in enumerate(substreams):
        print(f"\n📦 子流 {i+1}:")
        print(f"  - 事件数量: {len(sub.t)}")
        if len(sub.t) > 0:
            print(f"  - 第一个事件: [x, y, p, t] = [{sub.x[0]}, {sub.y[0]}, {sub.p[0]}, {sub.t[0]:.3f} ms]")
        else:
            print("  - ⚠️ 该时间段没有事件")

if __name__ == '__main__':
    main()