"""
CIFAR-10 DVS Dataset Academic Comparison Grid Generator
======================================================

This script generates academic-style comparison grids for CIFAR-10 DVS dataset.
CIFAR-10 DVS contains event-based versions of the classic CIFAR-10 object recognition dataset.

Usage:
    python cifar_comparison.py

Features:
- Multi-method comparison visualization
- Event density visualization with polarity separation
- LR upscaling support
- Customizable layout and colors
- Event filtering optimized for object recognition data
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from academic_comparison_grid import generate_academic_comparison_grid

# =============================================================================
# CIFAR-10 DVS DATASET CONFIGURATION
# =============================================================================

# Base path - modify this to your CIFAR-10 DVS dataset location
BASE_PATH = r"C:\Users\steve\Dataset\EVSR\cifar"

# Row configurations (different CIFAR-10 object categories)
ROW_CONFIGS = [
    # Different CIFAR-10 categories with sample files
    {"label": "(1)", "subpath": "airplane/1.npy"},  # Airplane category
    {"label": "(2)", "subpath": "automobile/15.npy"},  # Automobile category
    {"label": "(3)", "subpath": "bird/32.npy"},  # Bird category
    {"label": "(4)", "subpath": "cat/8.npy"},  # Cat category
]

# Column configurations (different methods)
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Baseline", "folder_path": "baseline/baseline-HRPre"},
    {"label": "(a)", "folder_path": "light/HRPre"},
    {"label": "(b)", "folder_path": "light-p-learn/HRPre"},
    {"label": "(c)", "folder_path": "Louck_light_p/HRPre"},
    {"label": "(d)", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Magnification bounding box for each row (x, y, width, height in pixels)
BBOX_CONFIGS = [
    {"x": 8, "y": 8, "width": 16, "height": 16},  # Row 1 - airplane
    {"x": 12, "y": 12, "width": 16, "height": 16},  # Row 2 - automobile
    {"x": 10, "y": 6, "width": 16, "height": 16},  # Row 3 - bird
    {"x": 14, "y": 10, "width": 16, "height": 16},  # Row 4 - cat
]

# Magnification settings for each row
MAGNIFY_CONFIGS = [
    {"position": "top-right", "scale": 3},  # Row 1
    {"position": "top-left", "scale": 3},  # Row 2
    {"position": "bottom-right", "scale": 3},  # Row 3
    {"position": "bottom-left", "scale": 3},  # Row 4
]

# Color settings for CIFAR-10 DVS event visualization
COLORS = {
    "positive": [1.0, 0.0, 0.0],  # Red for positive events
    "negative": [0.0, 0.0, 1.0],  # Blue for negative events
    "background": [1.0, 1.0, 1.0],  # White background
    "magnify": "Yellow",  # Yellow magnification border
}

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating CIFAR-10 DVS academic comparison grid...")
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
        bbox_configs=BBOX_CONFIGS,
        magnify_configs=MAGNIFY_CONFIGS,
        colors=COLORS,
        output_filename="cifar_academic_comparison.pdf",  # Change to .pdf for PDF export
        dpi=1000,
        figsize_per_cell=(2.5, 2.5),
        show_row_labels=True,
        show_column_labels=True,
        enable_magnification=False,  # 🔧 Set to False to disable magnification
        transpose_layout=False,  # 🔄 Set to True to swap rows and columns
        # 🎨 Event visualization - CIFAR-10 DVS OPTIMIZED FEATURES!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.5,  # Smoothing strength optimized for object recognition
        enhance_colors=True,  # Enhance color saturation and contrast
        # Event filtering optimized for CIFAR-10 object recognition
        event_sample_ratio=0.8,  # Use 80% of events (good balance for object details)
        time_window=None,  # Use all time, or try (0.1, 0.9) for middle portion
        polarity_separation=1.2,  # Moderate polarity separation for object features
        # 🎨 Layout customization - adjust these values as needed
        left_margin=0,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0,  # Bottom margin for column labels
        row_label_x=0.001,  # Row label X position (smaller = closer to edge)
        col_label_y=0.01,  # Column label Y distance from bottom (smaller = closer to bottom)
        row_label_fontsize=24,  # Row label font size
        col_label_fontsize=24,  # Column label font size
    )

    print("✅ CIFAR-10 DVS academic comparison grid generation completed!")
    print("📄 Output saved as: cifar_academic_comparison.pdf")
