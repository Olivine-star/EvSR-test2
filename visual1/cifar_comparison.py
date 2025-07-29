"""
CIFAR-10 DVS Dataset Academic Comparison Grid Generator
======================================================

This script generates academic-style comparison grids for CIFAR-10 DVS dataset.
CIFAR-10 DVS contains object recognition data captured with event cameras.

Usage:
    python cifar_comparison.py

Features:
- Multi-method comparison visualization
- Event density visualization with polarity separation
- LR upscaling support
- Customizable layout and colors
- Event filtering optimized for object recognition
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

# Row configurations (different CIFAR-10 object classes)
ROW_CONFIGS = [
    # Different object categories from CIFAR-10
    {"label": "(1)", "subpath": "airplane/3.npy"},  # Airplane class
    {"label": "(2)", "subpath": "automobile/3.npy"},  # Car class
    {"label": "(3)", "subpath": "bird/3.npy"},  # Bird class
    {"label": "(4)", "subpath": "cat/3.npy"},  # Cat class
]

# Column configurations (different methods)
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Baseline", "folder_path": "baseline/baseline-HRPre"},
    {"label": "Dual-Layer SNN", "folder_path": "light/HRPre"},
    {"label": "Dual-Layer SNN w/L", "folder_path": "light_p_learn/HRPre"},
    {"label": "Ultralight SNN", "folder_path": "Louck_light_p/HRPre"},
    {"label": "Ultralight SNN w/L", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Magnification bounding box for each row (x, y, width, height in pixels)
# Adjusted for CIFAR-10 DVS dataset dimensions (32x32 typical resolution)
BBOX_CONFIGS = [
    {"x": 15, "y": 8, "width": 12, "height": 12},  # Row 1 - center-right, upper
    {"x": 5, "y": 15, "width": 12, "height": 12},  # Row 2 - left, middle
    {"x": 20, "y": 20, "width": 10, "height": 10},  # Row 3 - right, lower
    {"x": 8, "y": 5, "width": 12, "height": 12},  # Row 4 - left, upper
]

# Magnification settings for each row
MAGNIFY_CONFIGS = [
    {"position": "bottom-left", "scale": 3.0},  # Row 1 - higher scale for small objects
    {"position": "top-right", "scale": 3.0},  # Row 2
    {"position": "top-left", "scale": 3.0},  # Row 3
    {"position": "bottom-right", "scale": 3.0},  # Row 4
]

# Color settings for event visualization - AGGRESSIVE density-based gradient colors
COLORS = {
    # AGGRESSIVE gradient colors: very light base that becomes very intense
    "positive": [1.0, 0.9, 0.9],  # Very light pink base for positive events
    "negative": [0.9, 0.9, 1.0],  # Very light blue base for negative events
    "positive_max": [1.0, 0.0, 0.0],  # Pure bright red for high density positive
    "negative_max": [0.0, 0.0, 1.0],  # Pure bright blue for high density negative
    "background": [1.0, 1.0, 1.0],  # White background
    "magnify": "Orange",  # Orange magnification border for object focus
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
        output_filename="cifar_academic_comparison.png",  # Change to .pdf for PDF export
        dpi=1000,
        figsize_per_cell=(2.5, 2.5),  # Square cells for CIFAR-10 (32x32 resolution)
        show_row_labels=True,
        show_column_labels=True,
        # 🔧 Magnification control - separate bounding box and magnification
        show_bounding_boxes=False,  # Show bounding box frames (True/False)
        enable_magnification=False,  # Show magnified insets (True/False)
        transpose_layout=False,  # 🔄 Set to True to swap rows and columns
        # 🎨 Event visualization - AGGRESSIVE CIFAR-10 DENSITY GRADIENT FEATURES!
        use_density=True,  # MUST be True for density-based gradient effect
        max_intensity=1.0,  # Full intensity range for maximum gradient visibility
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother gradients
        sigma=0.2,  # Reduced smoothing to preserve sharp gradient transitions
        enhance_colors=False,  # Disable color enhancement to preserve pure gradient
        # Event filtering optimized for AGGRESSIVE gradient visualization
        event_sample_ratio=1.0,  # Use all events for complete density information
        time_window=None,  # Use all time for complete motion capture
        polarity_separation=1.5,  # Enhanced separation for more distinct gradients
        # 🎨 Layout customization - adjust these values as needed
        left_margin=0,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0,  # Bottom margin for column labels
        row_label_x=0.001,  # Row label X position (smaller = closer to edge)
        col_label_y=0.01,  # Column label Y distance from bottom (smaller = closer to bottom)
        row_label_fontsize=24,  # Row label font size
        col_label_fontsize=18,  # Column label font size
    )

    print("✅ CIFAR-10 DVS academic comparison grid generation completed!")
    print("📄 Output saved as: cifar_academic_comparison.png")
    print("\n� AGGRESSIVE Density Gradient Features:")
    print("   • Very light pink → Pure bright red for positive events")
    print("   • Very light blue → Pure bright blue for negative events")
    print("   • Power function curve for enhanced contrast (x^0.5)")
    print("   • Aggressive polarity separation (1.5x)")
    print("   • Sharp gradient transitions (σ=0.2)")
    print("   • Enhanced mixed-polarity blending")
    print("\n�🔧 Magnification Control:")
    print("   • show_bounding_boxes=False: No bounding box frames (current)")
    print("   • enable_magnification=False: No magnified insets (current)")
    print("   • Can be controlled independently (box only, magnify only, or both)")
    print("\n📝 Next steps:")
    print(
        "1. Verify the file paths in ROW_CONFIGS and COLUMN_CONFIGS match your dataset structure"
    )
    print("2. Adjust BBOX_CONFIGS for optimal object region highlighting")
    print("3. Run the script to generate the comparison grid")
    print("4. Fine-tune visualization parameters for object recognition tasks")
