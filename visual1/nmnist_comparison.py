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
# Example structure: BASE_PATH/method_folder/digit/filename.npy
ROW_CONFIGS = [
    # Digit 0, file 0.npy
    {"label": "(1)", "subpath": "0/0.npy"},
    {"label": "(2)", "subpath": "2/1.npy"},
    {"label": "(3)", "subpath": "7/1.npy"},
]

# Column configurations - each column represents a different method
# folder_path is the path from BASE_PATH to the method folder
COLUMN_CONFIGS = [
    {"label": "LR", "folder_path": "SR_Test/SR_Test/LR"},
    {"label": "HR-GT", "folder_path": "SR_Test/SR_Test/HR"},
    {"label": "Li et al. (baseline)", "folder_path": "baseline/HRPre"},
    # Dual-Layer SNN with Learnable Loss (light-p-learn)
    {"label": "Ours (light-p-learn)", "folder_path": "light-p-learn/HRPre"},
    # Ultralight SNN (louck-light-p-learn)
    {"label": "Ours (louck-light-p-learn)", "folder_path": "Louck_light_p_learn/HRPre"},
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
        output_filename="nmnist_academic_comparison.png",
        dpi=300,
        figsize_per_cell=(2.5, 2.5),
        show_row_labels=True,
        show_column_labels=True,
        enable_magnification=False,  # 🔧 Set to True to enable magnification
        # 🎨 Event visualization - NEW ENHANCED FEATURES!
        use_density=True,  # Show event density (True) vs binary colors (False)
        max_intensity=1.0,  # Maximum color intensity (0.0-1.0)
        upscale_columns=[0],  # Upscale LR column (index 0) for better comparison
        upscale_factor=2,  # 2x upscaling for LR to match HR resolution
        smooth_visualization=True,  # Apply Gaussian smoothing for smoother appearance
        sigma=0.5,  # Smoothing strength (higher = more smooth)
        enhance_colors=True,  # Enhance color saturation and contrast
        # 🔥 NEW: Event filtering to reduce purple mixing and show clearer polarity
        event_sample_ratio=1,  # Use only 30% of events to reduce mixing
        time_window=(0.0, 1, 0),  # Use all time, or try (0.0, 0.5) for first half
        polarity_separation=1.5,  # Enhance polarity separation (1.0=normal, 2.0=max)
        # 🎨 Layout customization - adjust these values as needed
        wspace=0.01,  # Width spacing between images (smaller = more compact)
        hspace=0.01,  # Height spacing between images (smaller = more compact)
        left_margin=0.03,  # Left margin for row labels (smaller = labels closer to edge)
        bottom_margin=0.12,  # Bottom margin for column labels
        row_label_x=0.01,  # Row label X position (smaller = closer to edge)
        row_label_fontsize=12,  # Row label font size
        col_label_fontsize=12,  # Column label font size
        col_label_pad=10,  # Column label padding from image
    )

    print("✅ NMNIST academic comparison grid generation completed!")
    print("📸 Output saved as: nmnist_academic_comparison.png")
