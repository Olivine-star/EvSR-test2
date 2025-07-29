"""
NMNIST Dataset Academic Comparison Grid Example
Based on the actual NMNIST file structure with three-level paths
"""

from academic_comparison_grid import generate_academic_comparison_grid

# =============================================================================
# NMNIST DATASET CONFIGURATION
# =============================================================================

# Base path containing all your NMNIST data folders
BASE_PATH = r"C:\Users\steve\Dataset\EVSR\nmnist"

# Row configurations - each row represents a different sample
# subpath includes the digit folder and filename: "digit/filename.npy"
# polarity: "both" (default dual-polarity), "positive" (red density), "negative" (blue density)
# Example structure: BASE_PATH/method_folder/digit/filename.npy
ROW_CONFIGS = [
    # Digit 0, file 0.npy - HAS polarity="both" = use density method with Blues colormap
    {"label": "Pos", "subpath": "0/0.npy", "polarity": "positive"},
    {"label": "Neg", "subpath": "0/0.npy", "polarity": "negative"},
    {"label": "Pos", "subpath": "2/1.npy", "polarity": "positive"},
    {"label": "Neg", "subpath": "2/1.npy", "polarity": "negative"},
    {"label": "Pos", "subpath": "7/1.npy", "polarity": "positive"},
    {"label": "Neg", "subpath": "7/1.npy", "polarity": "negative"},
    {"label": "Pos", "subpath": "9/3.npy", "polarity": "positive"},
    {"label": "Neg", "subpath": "9/3.npy", "polarity": "negative"},
]

# Column configurations - each column represents a different method
# folder_path is the path from BASE_PATH to the method folder
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "baseline", "folder_path": "baseline/HRPre"},
    # Dual-Layer SNN with Learnable Loss (light-p-learn)
    {"label": "(a)", "folder_path": "light-p-learn/HRPre"},
    # Ultralight SNN (louck-light-p-learn)
    {"label": "(b)", "folder_path": "Louck_light_p_learn/HRPre"},
]

# Magnification bounding box for each row (x, y, width, height in pixels)
BBOX_CONFIGS = [
    {"x": 15, "y": 15, "width": 25, "height": 25},  # Row 1
    {"x": 20, "y": 10, "width": 25, "height": 25},  # Row 2
    {"x": 10, "y": 20, "width": 25, "height": 25},  # Row 3
    {"x": 25, "y": 15, "width": 25, "height": 25},  # Row 4
]

# Magnification settings for each row
MAGNIFY_CONFIGS = [
    {"position": "top-right", "scale": 2.5},  # Row 1
    {"position": "top-left", "scale": 2.5},  # Row 2
    {"position": "bottom-right", "scale": 2.5},  # Row 3
    {"position": "bottom-left", "scale": 2.5},  # Row 4
]

# Color settings for event visualization
COLORS = {
    "positive": [1.0, 0.0, 0.0],  # red for positive events
    "negative": [0.0, 0.0, 1.0],  # blue for negative events
    "background": [1.0, 1.0, 1.0],  # White background
    "magnify": "white",  # White magnification border
}

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Generating NMNIST academic comparison grid...")
    print(f"📁 Base path: {BASE_PATH}")
    print(f"📊 Grid size: {len(ROW_CONFIGS)} rows × {len(COLUMN_CONFIGS)} columns")
    print(f"📄 Sample subpaths: {[row['subpath'] for row in ROW_CONFIGS]}")

    # Example file paths that will be constructed:
    print("\n📂 Example file paths:")
    for i, row in enumerate(ROW_CONFIGS[:2]):  # Show first 2 examples
        for j, col in enumerate(COLUMN_CONFIGS[:3]):  # Show first 3 methods
            example_path = f"{BASE_PATH}\\{col['folder_path']}\\{row['subpath']}"
            print(f"   [{i + 1},{j + 1}]: {example_path}")
    print("   ...")

    # Generate the academic comparison grid
    fig, axes = generate_academic_comparison_grid(
        base_path=BASE_PATH,
        row_configs=ROW_CONFIGS,
        column_configs=COLUMN_CONFIGS,
        bbox_configs=BBOX_CONFIGS,
        magnify_configs=MAGNIFY_CONFIGS,
        colors=COLORS,
        output_filename="nmnist_academic_comparison.pdf",  # Change to .pdf for PDF export
        dpi=1000,
        figsize_per_cell=(2, 2),  # Much smaller cells to make spacing appear tighter
        show_row_labels=True,
        show_column_labels=True,
        # 🔧 Magnification control - separate bounding box and magnification
        show_bounding_boxes=False,  # Show bounding box frames (True/False)
        enable_magnification=False,  # Show magnified insets (True/False)
        transpose_layout=True,  # 🔄 Set to True to swap rows and columns
        # 🎨 Event visualization - NEW ENHANCED FEATURES!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=False,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.5,  # Smoothing strength (higher = more smooth)
        enhance_colors=True,  # Enhance color saturation and contrast
        # 🔥 NEW: Event filtering to reduce purple mixing and show clearer polarity
        event_sample_ratio=1,  # Use only 30% of events to reduce mixing
        time_window=(0.0, 1),  # Use all time, or try (0.0, 0.5) for first half
        polarity_separation=1.5,  # Enhance polarity separation (1.0=normal, 2.0=max)
        # 🎨 Layout customization - adjust these values as needed
        left_margin=0,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0,  # Bottom margin for column labels
        row_label_x=0.001,  # Row label X position (smaller = closer to edge)
        col_label_y=0.01,  # Column label Y distance from bottom (smaller = closer to bottom)
        row_label_fontsize=24,  # Row label font size
        col_label_fontsize=24,  # Column label font size
    )

    print("✅ NMNIST academic comparison grid generation completed!")
    print("� Output saved as: nmnist_academic_comparison.pdf")
    print("\n🔧 Magnification Control:")
    print("   • show_bounding_boxes=False: No bounding box frames (current)")
    print("   • enable_magnification=False: No magnified insets (current)")
    print("   • Can be controlled independently (box only, magnify only, or both)")
    print("\n📝 Configuration Options:")
    print("   🔧 For bounding boxes only:")
    print("      show_bounding_boxes=True, enable_magnification=False")
    print("   🔍 For magnification only:")
    print("      show_bounding_boxes=False, enable_magnification=True")
    print("   📦 For both:")
    print("      show_bounding_boxes=True, enable_magnification=True")
    print("   ❌ For neither (current):")
    print("      show_bounding_boxes=False, enable_magnification=False")
