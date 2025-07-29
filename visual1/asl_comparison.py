"""
ASL-DVS Dataset Academic Comparison Grid Generator
=================================================

This script generates academic-style comparison grids for ASL-DVS (American Sign Language) dataset.
ASL-DVS contains sign language gestures captured with event cameras.

Usage:
    python asl_comparison.py

Features:
- Multi-method comparison visualization
- Event density visualization with polarity separation
- LR upscaling support
- Customizable layout and colors
- Event filtering to reduce color mixing
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from academic_comparison_grid import generate_academic_comparison_grid

# =============================================================================
# ASL-DVS DATASET CONFIGURATION
# =============================================================================

# Base path - modify this to your ASL-DVS dataset location
BASE_PATH = r"C:\Users\steve\Dataset\EVSR\asl"

# Row configurations (different ASL signs/gestures)
ROW_CONFIGS = [
    # category a, file a_3880.npy
    # {"label": "(1)", "subpath": "y/y_0220.npy"},
    {"label": "(2)", "subpath": "v/v_0002.npy"},
    {"label": "(3)", "subpath": "h/h_4061.npy"},
    {"label": "(4)", "subpath": "o/o_4005.npy"},
]

# Column configurations (different methods)
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Baseline", "folder_path": "baseline/baseline-HRPre"},
    {"label": "Dual-Layer SNN", "folder_path": "light/HRPre"},
    {"label": "Dual-Layer SNN w/L", "folder_path": "light-p-learn/HRPre"},
    {"label": "Ultralight SNN", "folder_path": "Louck_light_p/HRPre"},
    {"label": "Ultralight SNN w/L", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Magnification bounding box for each row (x, y, width, height in pixels)
BBOX_CONFIGS = [
    #{"x": 110, "y": 70, "width": 30, "height": 30},  # Row 1
    {"x": 110, "y": 80, "width": 30, "height": 30},  # Row 2
    {"x": 140, "y": 30, "width": 30, "height": 30},  # Row 3
    {"x": 100, "y": 110, "width": 30, "height": 30},  # Row 4
]

# Magnification settings for each row
MAGNIFY_CONFIGS = [
    #{"position": "top-right", "scale": 2},  # Row 1
    {"position": "top-right", "scale": 2},  # Row 2
    {"position": "bottom-left", "scale": 2},  # Row 3
    {"position": "bottom-right", "scale": 2},  # Row 4
]

# Color settings for event visualization
COLORS = {
    "positive": [1.0, 0.0, 0.0],  # red for positive events
    "negative": [0.0, 0.0, 1.0],  # blue for negative events
    "background": [1.0, 1.0, 1.0],  # White background
    "magnify": "LightGreen",  # Light green magnification border
}
# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating ASL-DVS academic comparison grid...")
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
        output_filename="asl_academic_comparison.pdf",  # Change to .pdf for PDF export
        dpi=1000,
        figsize_per_cell=(2.5, 2.5),
        show_row_labels=True,
        show_column_labels=True,
        enable_magnification=True,  # 🔧 Set to False to disable magnification
        transpose_layout=False,  # 🔄 Set to True to swap rows and columns
        # 🎨 Event visualization - ASL-DVS OPTIMIZED FEATURES!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.5,  # Smoothing strength (higher = more smooth)
        enhance_colors=True,  # Enhance color saturation and contrast
        # Event filtering to reduce purple mixing and show clearer polarity
        event_sample_ratio=1,  # Use only 40% of events to reduce mixing
        time_window=None,  # Use all time, or try (0.0, 0.5) for first half
        polarity_separation=1,  # Enhance polarity separation (1.0=normal, 2.0=max)
        # 🎨 Layout customization - adjust these values as needed
        left_margin=0,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0,  # Bottom margin for column labels
        row_label_x=0.001,  # Row label X position (smaller = closer to edge)
        col_label_y=0.01,  # Column label Y distance from bottom (smaller = closer to bottom)
        row_label_fontsize=24,  # Row label font size
        col_label_fontsize=18,  # Column label font size
    )

    print("✅ ASL-DVS academic comparison grid generation completed!")
    print("📄 Output saved as: asl_academic_comparison.pdf")
