"""
Test All Datasets - Batch Generation Script
===========================================

This script tests all four dataset comparison generators:
- NMNIST (handwritten digits)
- ASL-DVS (sign language gestures)  
- IR (infrared/thermal data)
- NFS (high-speed automotive data)

Usage:
    python test_all_datasets.py

Note: Make sure to update the BASE_PATH in each dataset file to point to your actual data location.
"""

import os
import sys

def test_dataset(dataset_name, script_name):
    """Test a single dataset comparison generator"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {dataset_name} Dataset")
    print(f"{'='*60}")
    
    try:
        # Import and run the dataset script
        if script_name == "nmnist_comparison":
            import nmnist_comparison
        elif script_name == "asl_comparison":
            import asl_comparison
        elif script_name == "ir_comparison":
            import ir_comparison
        elif script_name == "nfs_comparison":
            import nfs_comparison
        
        print(f"✅ {dataset_name} test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"⚠️  {dataset_name} data files not found: {e}")
        print(f"   Please update BASE_PATH in {script_name}.py to point to your data location")
        
    except Exception as e:
        print(f"❌ {dataset_name} test failed: {e}")

def main():
    """Test all dataset generators"""
    print("🚀 Testing All Dataset Comparison Generators")
    print("=" * 60)
    
    datasets = [
        ("NMNIST", "nmnist_comparison"),
        ("ASL-DVS", "asl_comparison"), 
        ("IR", "ir_comparison"),
        ("NFS", "nfs_comparison"),
    ]
    
    for dataset_name, script_name in datasets:
        test_dataset(dataset_name, script_name)
    
    print(f"\n{'='*60}")
    print("🎯 All Dataset Tests Completed!")
    print("📁 Generated files:")
    
    output_files = [
        "nmnist_academic_comparison.png",
        "asl_academic_comparison.png", 
        "ir_academic_comparison.png",
        "nfs_academic_comparison.png",
    ]
    
    for filename in output_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (not generated)")
    
    print("\n💡 Tips:")
    print("   - Update BASE_PATH in each dataset file to your data location")
    print("   - Adjust ROW_CONFIGS and COLUMN_CONFIGS for your data structure")
    print("   - Modify event filtering parameters for optimal visualization")
    print("   - Each dataset has optimized color schemes and parameters")

if __name__ == "__main__":
    main()
