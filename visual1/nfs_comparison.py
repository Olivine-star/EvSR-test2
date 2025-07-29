"""
NFS Dataset Academic Comparison Grid Generator
==============================================

This script generates academic-style comparison grids for NFS (Need for Speed) dataset.
NFS contains high-speed motion sequences captured with event cameras.

Usage:
    python nfs_comparison.py

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
# NFS DATASET CONFIGURATION
# =============================================================================

# Base path - modify this to your NFS dataset location
BASE_PATH = (
    r"C:\code\EventSR-Project\EventSR-dataset\dataset\NFS\data-visual-test-redata-ef6"
)

# Row configurations (different NFS sequences/samples)
# Based on actual NFS dataset structure with sequences: 11, 19, 2, 4, 50, 65, 74, 80, 86, 99
ROW_CONFIGS = [
    # Select representative sequences and frames for comparison
    {
        "label": "(1)",
        "subpath": "10/210.npy",
    },  # Sequence 2, frame 7 (early high activity)
    {"label": "(2)", "subpath": "35/84.npy"},  # Sequence 4, frame 25 (mid activity)
    {
        "label": "(3)",
        "subpath": "85/200.npy",
    },  # Sequence 50, frame 50 (varied activity)
]

# Column configurations (different methods)
# Based on actual NFS dataset folder structure
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "LR"},  # Low resolution input
    {"label": "HR-GT", "folder_path": "HR"},  # High resolution ground truth
    {
        "label": "Baseline",
        "folder_path": "NFS-base(50ms-redata)-non-leak",
    },  # Baseline model results
    {
        "label": "Dual-Layer SNN",
        "folder_path": "NFS-light(50ms-redata)-leak",
    },  # Light model
    {
        "label": "Dual-Layer SNN w/L",
        "folder_path": "NFS-light-p-learn-(50ms-redata)-leak",
    },  # Light model with learnable params
    {
        "label": "Ultralight SNN",
        "folder_path": "NFS-Louck-light-p(50ms-redata)-leak",
    },  # Louck light model with params
    {
        "label": "Ultralight SNN w/L",
        "folder_path": "NFS-Louck-light-p-learn(50ms-redata)-leak",
    },  # Louck light model with learnable params
]

# Magnification bounding box for each row (x, y, width, height in pixels)
# Adjusted for NFS dataset dimensions (124x222) with origin="lower" coordinate system
# Y coordinates are now measured from bottom (0) to top (124)
BBOX_CONFIGS = [
    {
        "x": 120,
        "y": 50,
        "width": 30,
        "height": 30,
    },  # Row 1 - right side, upper (124-30-25=69)
    {
        "x": 110,
        "y": 90,
        "width": 40,
        "height": 25,
    },  # Row 2 - left side, middle (124-50-25=49)
    {
        "x": 75,
        "y": 20,
        "width": 95,
        "height": 85,
    },  # Row 3 - right side, lower (124-70-25=29)
]

# Magnification settings for each row
MAGNIFY_CONFIGS = [
    {"position": "bottom-left", "scale": 2.5},  # Row 1
    {"position": "top-right", "scale": 2.5},  # Row 2
    {"position": "top-left", "scale": 2.5},  # Row 3
    {"position": "bottom-right", "scale": 2.5},  # Row 4
]

# Color settings for event visualization - AGGRESSIVE density-based gradient colors
COLORS = {
    # AGGRESSIVE gradient colors: very light base that becomes very intense
    "positive": [1.0, 0.9, 0.9],  # Very light pink base for positive events
    "negative": [0.9, 0.9, 1.0],  # Very light blue base for negative events
    "positive_max": [1.0, 0.0, 0.0],  # Pure bright red for high density positive
    "negative_max": [0.0, 0.0, 1.0],  # Pure bright blue for high density negative
    "background": [1.0, 1.0, 1.0],  # White background
    "magnify": "Green",  # Yellow magnification border
}

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating NFS academic comparison grid...")
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
        output_filename="nfs_academic_comparison.pdf",  # Change to .pdf for PDF export
        dpi=1000,
        figsize_per_cell=(3.58, 2),  # NFS aspect ratio: 222×124 ≈ 1.79:1
        show_row_labels=True,
        show_column_labels=True,
        # 🔧 Magnification control - separate bounding box and magnification
        show_bounding_boxes=True,  # Show bounding box frames (True/False)
        enable_magnification=False,  # Show magnified insets (True/False)
        transpose_layout=False,  # 🔄 Set to True to swap rows and columns
        # 🎨 Event visualization - AGGRESSIVE NFS DENSITY GRADIENT FEATURES!
        use_density=True,  # MUST be True for density-based gradient effect
        max_intensity=1.0,  # Full intensity range for maximum gradient visibility
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother gradients
        sigma=0.3,  # Reduced smoothing to preserve sharp gradient transitions
        enhance_colors=False,  # Disable color enhancement to preserve pure gradient
        # Event filtering optimized for AGGRESSIVE gradient visualization
        event_sample_ratio=1.0,  # Use all events for complete density information
        time_window=None,  # Use all time for complete motion capture
        polarity_separation=1.0,  # Enhanced separation for more distinct gradients
        # 🎨 Layout customization - adjust these values as needed
        left_margin=0,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0,  # Bottom margin for column labels
        row_label_x=0.001,  # Row label X position (smaller = closer to edge)
        col_label_y=0.01,  # Column label Y distance from bottom (smaller = closer to bottom)
        row_label_fontsize=24,  # Row label font size
        col_label_fontsize=24,  # Column label font size
    )

    print("✅ NFS academic comparison grid generation completed!")
    print("📄 Output saved as: nfs_academic_comparison.png")
    print("\n� Image Aspect Ratio:")
    print("   • Proper NFS dimensions: 124×222 (aspect ratio 1.79:1)")
    print("   • Cell size: 4.5×2.5 to maintain rectangular shape")
    print("   • No forced square distortion")
    print("   • Fixed coordinate system: bounding boxes now match magnified regions")
    print("\n🔧 Magnification Control:")
    print("   • show_bounding_boxes=True: Show yellow bounding box frames")
    print("   • enable_magnification=True: Show magnified inset windows")
    print("   • Can be controlled independently (box only, magnify only, or both)")
    print("\n�🔥 AGGRESSIVE Density Gradient Features:")
    print("   • Very light pink → Pure bright red for positive events")
    print("   • Very light blue → Pure bright blue for negative events")
    print("   • Power function curve for enhanced contrast (x^0.5)")
    print("   • Aggressive polarity separation (1.5x)")
    print("   • Sharp gradient transitions (σ=0.2)")
    print("   • Enhanced mixed-polarity blending")
    print("\n📝 Configuration Options:")
    print("   🔧 For bounding boxes only:")
    print("      show_bounding_boxes=True, enable_magnification=False")
    print("   🔍 For magnification only:")
    print("      show_bounding_boxes=False, enable_magnification=True")
    print("   📦 For both (current setting):")
    print("      show_bounding_boxes=True, enable_magnification=True")
    print("\n📝 Next steps:")
    print(
        "1. Verify the file paths in ROW_CONFIGS and COLUMN_CONFIGS match your dataset structure"
    )
    print("2. Adjust BBOX_CONFIGS for optimal magnification areas")
    print("3. Run the script to generate the comparison grid")
    print("4. Fine-tune visualization parameters if needed")
