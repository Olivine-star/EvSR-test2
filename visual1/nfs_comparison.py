"""
NFS Dataset Academic Comparison Grid Generator
==============================================

This script generates academic-style comparison grids for NFS (Need for Speed) dataset.
NFS dataset contains high-speed automotive event camera data for motion analysis.

Usage:
    python nfs_comparison.py

Features:
- Multi-method comparison visualization
- Event density visualization with polarity separation
- LR upscaling support
- Customizable layout and colors
- Event filtering optimized for high-speed motion data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from academic_comparison_grid import generate_academic_comparison_grid

# =============================================================================
# NFS DATASET CONFIGURATION
# =============================================================================

# Row configurations (different NFS scenes/tracks)
ROW_CONFIGS = [
    {"label": "(1)", "subpath": "track1/0.npy"},  # Track 1, file 0.npy
    {"label": "(2)", "subpath": "track2/1.npy"},  # Track 2, file 1.npy
    {"label": "(3)", "subpath": "track3/2.npy"},  # Track 3, file 2.npy
]

# Column configurations (different methods)
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Baseline", "folder_path": "baseline/HRPre"},
    {"label": "Light-P-Learn", "folder_path": "light-p-learn/HRPre"},
    {"label": "Louck-Light-P", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Base path - modify this to your NFS dataset location
BASE_PATH = r"C:\Users\steve\Downloads\nfs"

# Color scheme for NFS event visualization (optimized for motion data)
COLORS = {
    "positive": [0.0, 1.0, 0.0],  # Green for positive events (forward motion)
    "negative": [1.0, 0.0, 1.0],  # Magenta for negative events (backward motion)
    "background": [0.0, 0.0, 0.0],  # Black background
    "magnify": "white",
}

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating NFS dataset academic comparison grid...")
    print(f"📁 Base path: {BASE_PATH}")
    print(f"📊 Grid size: {len(ROW_CONFIGS)} rows × {len(COLUMN_CONFIGS)} columns")
    print(f"📄 Sample subpaths: {[row['subpath'] for row in ROW_CONFIGS]}")
    
    # Show example file paths for verification
    print("\n📂 Example file paths:")
    for i, row in enumerate(ROW_CONFIGS[:3], 1):
        for j, col in enumerate(COLUMN_CONFIGS[:3], 1):
            example_path = os.path.join(BASE_PATH, col["folder_path"], row["subpath"])
            print(f"   [{i},{j}]: {example_path}")
        if i < len(ROW_CONFIGS[:3]):
            print("   ...")
    
    # Generate the comparison grid
    fig, axes = generate_academic_comparison_grid(
        base_path=BASE_PATH,
        row_configs=ROW_CONFIGS,
        column_configs=COLUMN_CONFIGS,
        colors=COLORS,
        output_filename="nfs_academic_comparison.png",
        dpi=300,
        figsize_per_cell=(2.5, 2.5),
        show_row_labels=True,
        show_column_labels=True,
        enable_magnification=False,  # 🔧 Set to False to disable magnification
        # 🎨 Event visualization - NFS HIGH-SPEED MOTION OPTIMIZED!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.3,  # Lower smoothing to preserve motion details
        enhance_colors=True,  # Enhance color saturation and contrast
        # 🔥 NEW: Event filtering optimized for high-speed motion
        event_sample_ratio=0.2,  # Use only 20% of events (high-speed data is very dense)
        time_window=(0.1, 0.9),  # Skip first/last 10% to avoid edge effects
        polarity_separation=1.8,  # High separation to distinguish motion directions
        # 🎨 Layout customization - adjust these values as needed
        wspace=0.01,  # Width spacing between images (smaller = more compact)
        hspace=0.01,  # Height spacing between images (smaller = more compact)
        left_margin=0.02,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0.12,  # Bottom margin for column labels
        tight_layout_ad=0.5,  # Overall padding
        row_label_x=0.01,  # Row label X position (smaller = closer to edge)
        row_label_fontsize=12,  # Row label font size
        col_label_fontsize=12,  # Column label font size
        col_label_pad=10,  # Column label padding from image
    )

    print("✅ NFS dataset academic comparison grid generation completed!")
    print("📸 Output saved as: nfs_academic_comparison.png")
