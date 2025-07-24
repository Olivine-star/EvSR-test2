"""
IR Dataset Academic Comparison Grid Generator
============================================

This script generates academic-style comparison grids for IR (Infrared) event camera dataset.
IR dataset contains thermal/infrared event data for various scenarios.

Usage:
    python ir_comparison.py

Features:
- Multi-method comparison visualization
- Event density visualization with polarity separation
- LR upscaling support
- Customizable layout and colors
- Event filtering optimized for IR data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from academic_comparison_grid import generate_academic_comparison_grid

# =============================================================================
# IR DATASET CONFIGURATION
# =============================================================================

# Row configurations (different IR scenes/objects)
ROW_CONFIGS = [
    {"label": "(1)", "subpath": "scene1/0.npy"},  # IR scene 1, file 0.npy
    {"label": "(2)", "subpath": "scene2/1.npy"},  # IR scene 2, file 1.npy
    {"label": "(3)", "subpath": "scene3/2.npy"},  # IR scene 3, file 2.npy
]

# Column configurations (different methods)
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Baseline", "folder_path": "baseline/HRPre"},
    {"label": "Light-P-Learn", "folder_path": "light-p-learn/HRPre"},
    {"label": "Louck-Light-P", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Base path - modify this to your IR dataset location
BASE_PATH = r"C:\Users\steve\Dataset\EVSR\ir"

# Color scheme for IR event visualization (optimized for thermal data)
COLORS = {
    "positive": [1.0, 0.5, 0.0],  # Orange for positive events (hot)
    "negative": [0.0, 0.5, 1.0],  # Light blue for negative events (cold)
    "background": [0.0, 0.0, 0.0],  # Black background
    "magnify": "white",
}

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating IR dataset academic comparison grid...")
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
        output_filename="ir_academic_comparison.png",
        dpi=300,
        figsize_per_cell=(2.5, 2.5),
        show_row_labels=True,
        show_column_labels=True,
        enable_magnification=False,  # 🔧 Set to False to disable magnification
        # 🎨 Event visualization - IR OPTIMIZED FEATURES!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.8,  # Higher smoothing for IR data (thermal gradients)
        enhance_colors=True,  # Enhance color saturation and contrast
        # 🔥 NEW: Event filtering optimized for IR thermal data
        event_sample_ratio=0.5,  # Use 50% of events (IR data often has good SNR)
        time_window=None,  # Use all time, or try (0.2, 0.8) for middle portion
        polarity_separation=1.2,  # Moderate separation for thermal gradients
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

    print("✅ IR dataset academic comparison grid generation completed!")
    print("📸 Output saved as: ir_academic_comparison.png")
