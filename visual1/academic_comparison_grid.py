import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os


def gaussian_smooth_2d(image, sigma=1.0):
    """Apply Gaussian smoothing to 2D image using numpy"""
    if sigma <= 0:
        return image

    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    kernel = kernel / np.sum(kernel)

    # Apply convolution
    padded = np.pad(image, center, mode="constant")
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(
                padded[i : i + kernel_size, j : j + kernel_size] * kernel
            )

    return result


def enhance_colors_func(rgb_image, enhancement_factor=1.5, gamma=0.8):
    """Enhance color saturation and apply gamma correction"""
    # Apply gamma correction for better contrast
    rgb_image = np.power(rgb_image, gamma)

    # Enhance saturation
    # Convert to HSV-like enhancement
    max_val = np.max(rgb_image, axis=2, keepdims=True)
    min_val = np.min(rgb_image, axis=2, keepdims=True)

    # Create saturation mask (expand to match RGB dimensions)
    saturation_mask = (max_val > 0.01).squeeze(axis=2)

    # Enhance saturation
    enhanced = rgb_image.copy()

    # Apply enhancement to each channel separately
    for channel in range(3):
        mask = saturation_mask
        enhanced[mask, channel] = (
            min_val[mask, 0]
            + (rgb_image[mask, channel] - min_val[mask, 0]) * enhancement_factor
        )

    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


def create_event_visualization(
    events,
    img_size,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],  # Red for positive
    negative_color=[0.0, 0.0, 1.0],  # Blue for negative
    background_color=[0.0, 0.0, 0.0],  # Black background
    use_density=True,  # Whether to show event density
    max_intensity=1.0,  # Maximum color intensity
    upscale_factor=1,  # Upscaling factor for LR images
    smooth_visualization=True,  # Apply Gaussian smoothing for better appearance
    sigma=0.8,  # Gaussian smoothing parameter
    enhance_colors=True,  # Enhance color saturation
    # 🔥 NEW: Event sampling parameters
    event_sample_ratio=1.0,  # Ratio of events to use (0.0-1.0)
    time_window=None,  # (start_ratio, end_ratio) to select time window
    polarity_separation=1.0,  # How much to separate positive/negative colors (0.0-2.0)
):
    """Create event visualization with density information and red-blue polarity"""

    # 🔥 NEW: Apply event filtering
    if event_sample_ratio < 1.0 or time_window is not None:
        # Time window filtering
        if time_window is not None and len(events) > 0:
            timestamps = [event[0] for event in events]
            min_ts, max_ts = min(timestamps), max(timestamps)
            ts_range = max_ts - min_ts

            start_ts = min_ts + time_window[0] * ts_range
            end_ts = min_ts + time_window[1] * ts_range

            events = [event for event in events if start_ts <= event[0] <= end_ts]

        # Event sampling
        if event_sample_ratio < 1.0 and len(events) > 0:
            import random

            sample_size = int(len(events) * event_sample_ratio)
            events = random.sample(list(events), sample_size)

    # Apply upscaling if needed
    if upscale_factor > 1:
        # Scale up the target image size
        upscaled_size = (img_size[0] * upscale_factor, img_size[1] * upscale_factor)
        positive_counts = np.zeros(upscaled_size, dtype=int)
        negative_counts = np.zeros(upscaled_size, dtype=int)

        # Count events with upscaling
        for event in events:
            _, x, y, p = event  # timestamp not used
            # Scale coordinates
            x_scaled = int(x * upscale_factor)
            y_scaled = int(y * upscale_factor)

            # Check bounds for upscaled image
            if 0 <= y_scaled < upscaled_size[0] and 0 <= x_scaled < upscaled_size[1]:
                if (
                    polarity_filter == "both"
                    or (polarity_filter == "positive" and p > 0)
                    or (polarity_filter == "negative" and p <= 0)
                ):
                    # Fill upscaled block (nearest neighbor upsampling)
                    for dy in range(upscale_factor):
                        for dx in range(upscale_factor):
                            new_y = y_scaled + dy
                            new_x = x_scaled + dx
                            if new_y < upscaled_size[0] and new_x < upscaled_size[1]:
                                if p > 0:  # Positive polarity
                                    positive_counts[new_y, new_x] += 1
                                else:  # Negative polarity
                                    negative_counts[new_y, new_x] += 1

        # Update img_size to upscaled size
        img_size = upscaled_size
    else:
        # Original algorithm for non-upscaled images
        positive_counts = np.zeros(img_size, dtype=int)
        negative_counts = np.zeros(img_size, dtype=int)

        # Count events for each polarity
        for event in events:
            _, x, y, p = event  # timestamp not used
            x_int, y_int = int(x), int(y)

            # Check bounds
            if 0 <= y_int < img_size[0] and 0 <= x_int < img_size[1]:
                if (
                    polarity_filter == "both"
                    or (polarity_filter == "positive" and p > 0)
                    or (polarity_filter == "negative" and p <= 0)
                ):
                    if p > 0:  # Positive polarity
                        positive_counts[y_int, x_int] += 1
                    else:  # Negative polarity
                        negative_counts[y_int, x_int] += 1

    if use_density:
        # Create density-based visualization
        rgb_image = create_density_visualization(
            positive_counts,
            negative_counts,
            img_size,
            positive_color,
            negative_color,
            background_color,
            max_intensity,
            smooth_visualization,
            sigma,
            polarity_separation,
        )
    else:
        # Create simple binary visualization (original method)
        rgb_image = create_binary_visualization(
            positive_counts,
            negative_counts,
            img_size,
            positive_color,
            negative_color,
            background_color,
        )

    # Apply color enhancement if requested
    if enhance_colors:
        rgb_image = enhance_colors_func(rgb_image, enhancement_factor=1.5, gamma=0.7)

    return rgb_image


def create_density_visualization(
    positive_counts,
    negative_counts,
    img_size,
    positive_color,
    negative_color,
    background_color,
    max_intensity,
    smooth_visualization=True,
    sigma=0.8,
    polarity_separation=1.0,
):
    """Create visualization showing event density with color intensity"""
    # Initialize RGB image
    rgb_image = np.full(
        (img_size[0], img_size[1], 3), background_color, dtype=np.float32
    )

    # Find maximum counts for normalization
    max_pos = np.max(positive_counts) if np.max(positive_counts) > 0 else 1
    max_neg = np.max(negative_counts) if np.max(negative_counts) > 0 else 1

    # Process each pixel
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            pos_count = positive_counts[y, x]
            neg_count = negative_counts[y, x]

            if pos_count > 0 and neg_count > 0:
                # Mixed polarity: blend colors based on ratio and separation
                total_count = pos_count + neg_count
                pos_ratio = pos_count / total_count
                neg_ratio = neg_count / total_count

                # Apply polarity separation (higher values make colors more distinct)
                if polarity_separation != 1.0:
                    # Enhance the dominant polarity
                    if pos_ratio > neg_ratio:
                        pos_ratio = min(pos_ratio * polarity_separation, 1.0)
                        neg_ratio = 1.0 - pos_ratio
                    else:
                        neg_ratio = min(neg_ratio * polarity_separation, 1.0)
                        pos_ratio = 1.0 - neg_ratio

                # Normalize intensity based on total events
                intensity = (
                    min(total_count / max(max_pos, max_neg), 1.0) * max_intensity
                )

                # Blend colors
                rgb_image[y, x] = (
                    pos_ratio * np.array(positive_color) * intensity
                    + neg_ratio * np.array(negative_color) * intensity
                )
            elif pos_count > 0:
                # Only positive events
                intensity = min(pos_count / max_pos, 1.0) * max_intensity
                rgb_image[y, x] = np.array(positive_color) * intensity
            elif neg_count > 0:
                # Only negative events
                intensity = min(neg_count / max_neg, 1.0) * max_intensity
                rgb_image[y, x] = np.array(negative_color) * intensity

    # Apply Gaussian smoothing if requested
    if smooth_visualization and sigma > 0:
        # Smooth each color channel separately
        for channel in range(3):
            rgb_image[:, :, channel] = gaussian_smooth_2d(
                rgb_image[:, :, channel], sigma
            )

    return rgb_image


def create_binary_visualization(
    positive_counts,
    negative_counts,
    img_size,
    positive_color,
    negative_color,
    background_color,
):
    """Create simple binary visualization (original method)"""
    rgb_image = np.full(
        (img_size[0], img_size[1], 3), background_color, dtype=np.float32
    )

    for y in range(img_size[0]):
        for x in range(img_size[1]):
            pos_count = positive_counts[y, x]
            neg_count = negative_counts[y, x]

            if pos_count > 0 and neg_count > 0:
                # Mixed polarity: use the dominant one
                if pos_count >= neg_count:
                    rgb_image[y, x] = positive_color
                else:
                    rgb_image[y, x] = negative_color
            elif pos_count > 0:
                rgb_image[y, x] = positive_color
            elif neg_count > 0:
                rgb_image[y, x] = negative_color

    return rgb_image


def add_magnified_inset(
    ax,
    image,
    bbox_x,
    bbox_y,
    bbox_width,
    bbox_height,
    magnify_position="top-right",
    magnify_scale=2.0,
    magnify_color="white",
    magnify_linewidth=2.0,
    upscale_factor=1,
):
    """Add magnified inset with white border"""
    img_height, img_width = image.shape[:2]

    # Ensure bbox is within bounds
    bbox_x = max(0, min(bbox_x, img_width - bbox_width))
    bbox_y = max(0, min(bbox_y, img_height - bbox_height))
    bbox_width = min(bbox_width, img_width - bbox_x)
    bbox_height = min(bbox_height, img_height - bbox_y)

    # Extract magnified region
    magnified_region = image[
        bbox_y : bbox_y + bbox_height, bbox_x : bbox_x + bbox_width
    ]

    if magnified_region.size == 0:
        return

    # Add bounding box
    rect = Rectangle(
        (bbox_x, bbox_y),
        bbox_width,
        bbox_height,
        linewidth=magnify_linewidth,
        edgecolor=magnify_color,
        facecolor="none",
    )
    ax.add_patch(rect)

    # Calculate inset position
    inset_width = int(bbox_width * magnify_scale)
    inset_height = int(bbox_height * magnify_scale)
    margin = max(3, min(10, img_width // 15, img_height // 15))

    # Position inset
    if magnify_position == "top-right":
        inset_x = img_width - inset_width - margin
        inset_y = img_height - inset_height - margin
    elif magnify_position == "top-left":
        inset_x = margin
        inset_y = img_height - inset_height - margin
    elif magnify_position == "bottom-right":
        inset_x = img_width - inset_width - margin
        inset_y = margin
    elif magnify_position == "bottom-left":
        inset_x = margin
        inset_y = margin
    else:
        inset_x = img_width - inset_width - margin
        inset_y = img_height - inset_height - margin

    # Ensure inset is within bounds
    inset_x = max(0, min(inset_x, img_width - inset_width))
    inset_y = max(0, min(inset_y, img_height - inset_height))

    # Create inset axes
    inset_ax = ax.inset_axes(
        [
            inset_x / img_width,
            inset_y / img_height,
            inset_width / img_width,
            inset_height / img_height,
        ]
    )

    # Display magnified region
    inset_ax.imshow(magnified_region, aspect="equal", interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    # Add white border
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(magnify_color)
        spine.set_linewidth(magnify_linewidth)

    return inset_ax


def generate_academic_comparison_grid(
    base_path,
    row_configs,
    column_configs,
    bbox_configs=None,
    magnify_configs=None,
    colors=None,
    output_filename="academic_comparison_grid.png",
    dpi=300,
    figsize_per_cell=(3, 3),
    show_row_labels=True,
    show_column_labels=True,
    enable_magnification=True,
    # Event visualization parameters
    use_density=True,  # Whether to show event density (True) or binary (False)
    max_intensity=1.0,  # Maximum color intensity for density visualization
    upscale_columns=None,  # List of column indices to upscale (e.g., [0] for LR)
    upscale_factor=2,  # Upscaling factor for specified columns
    smooth_visualization=True,  # Apply Gaussian smoothing for better appearance
    sigma=0.8,  # Gaussian smoothing parameter
    enhance_colors=True,  # Enhance color saturation and contrast
    # 🔥 NEW: Event filtering parameters
    event_sample_ratio=1.0,  # Ratio of events to use (0.0-1.0, e.g., 0.5 for 50%)
    time_window=None,  # (start_ratio, end_ratio) to select time window, e.g., (0.0, 0.5)
    polarity_separation=1.0,  # How much to separate positive/negative colors (0.5-2.0)
    # Layout customization parameters
    wspace=0.01,  # Width spacing between subplots
    hspace=0.01,  # Height spacing between subplots
    left_margin=0.05,  # Left margin for row labels
    bottom_margin=0.12,  # Bottom margin for column labels
    row_label_x=0.01,  # X position of row labels (0-1)
    row_label_fontsize=12,  # Font size for row labels
    col_label_fontsize=12,  # Font size for column labels
    col_label_pad=10,  # Padding for column labels
):
    """
    Generate academic comparison grid visualization

    Parameters:
    - base_path: Base directory path containing data folders
    - row_configs: List of dicts with row info: [{"label": "(1)", "subpath": "3/9.npy"}, ...]
    - column_configs: List of dicts with column info: [{"label": "LR", "folder_path": "light-p-learn/HRPre"}, ...]
    - bbox_configs: List of bbox configs for each row: [{"x": 10, "y": 10, "width": 20, "height": 20}, ...]
    - magnify_configs: Magnification settings for each row
    - colors: Color settings dict
    - output_filename: Output filename
    - dpi: Output resolution
    - figsize_per_cell: Size of each cell (width, height)
    - show_row_labels: Whether to show row labels
    - show_column_labels: Whether to show column labels
    - enable_magnification: Whether to enable magnification insets (True/False)

    Event visualization:
    - use_density: Show event density (True) or binary visualization (False) (default: True)
    - max_intensity: Maximum color intensity for density visualization (default: 1.0)
    - upscale_columns: List of column indices to upscale (e.g., [0] for LR column) (default: None)
    - upscale_factor: Upscaling factor for specified columns (default: 2)
    - smooth_visualization: Apply Gaussian smoothing for better appearance (default: True)
    - sigma: Gaussian smoothing parameter, higher = more smooth (default: 0.8)
    - enhance_colors: Enhance color saturation and contrast (default: True)

    Layout customization:
    - wspace: Width spacing between subplots (default: 0.01)
    - hspace: Height spacing between subplots (default: 0.01)
    - left_margin: Left margin for row labels (default: 0.05)
    - bottom_margin: Bottom margin for column labels (default: 0.12)
    - tight_layout_pad: Padding for tight_layout (default: 0.5)
    - row_label_x: X position of row labels 0-1 (default: 0.01)
    - row_label_fontsize: Font size for row labels (default: 12)
    - col_label_fontsize: Font size for column labels (default: 12)
    - col_label_pad: Padding for column labels (default: 10)

    File path construction: base_path + column_folder_path + row_subpath
    Example: C:/data + light-p-learn/HRPre + 3/9.npy = C:/data/light-p-learn/HRPre/3/9.npy
    """

    # Default colors
    if colors is None:
        colors = {
            "positive": [1.0, 0.0, 0.0],  # Red
            "negative": [0.0, 0.0, 1.0],  # Blue
            "background": [0.0, 0.0, 0.0],  # Black
            "magnify": "white",
        }

    # Default bbox configs
    if bbox_configs is None:
        bbox_configs = [
            {"x": 10, "y": 10, "width": 20, "height": 20} for _ in row_configs
        ]

    # Default magnify configs
    if magnify_configs is None:
        magnify_configs = [{"position": "top-right", "scale": 2.0} for _ in row_configs]

    # Calculate figure size
    n_rows = len(row_configs)
    n_cols = len(column_configs)
    figsize = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)

    # Create figure with specified spacing
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        gridspec_kw={"hspace": hspace, "wspace": wspace},
    )
    fig.patch.set_facecolor(colors["background"])

    # Ensure axes is 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    print(f"Creating {n_rows}x{n_cols} academic comparison grid...")

    # Find the maximum image dimensions for consistent sizing
    max_height, max_width = 0, 0
    temp_images = {}

    # First pass: load all images and find max dimensions (considering upscaling)
    for row_idx, row_config in enumerate(row_configs):
        for col_idx, col_config in enumerate(column_configs):
            file_path = os.path.join(
                base_path, col_config["folder_path"], row_config["subpath"]
            )
            if os.path.exists(file_path):
                events = np.load(file_path)
                max_x = int(np.max(events[:, 1])) + 1
                max_y = int(np.max(events[:, 2])) + 1

                # Apply upscaling factor if this column should be upscaled
                if upscale_columns and col_idx in upscale_columns:
                    max_x *= upscale_factor
                    max_y *= upscale_factor

                max_width = max(max_width, max_x)
                max_height = max(max_height, max_y)
                temp_images[(row_idx, col_idx)] = (events, (max_y, max_x))

    print(f"📐 Unified image size: {max_width} x {max_height}")
    if upscale_columns:
        print(f"🔍 Upscaling columns {upscale_columns} by {upscale_factor}x")

    # Process each cell
    for row_idx, row_config in enumerate(row_configs):
        for col_idx, col_config in enumerate(column_configs):
            ax = axes[row_idx, col_idx]

            # Construct file path: base_path + column_folder_path + row_subpath
            file_path = os.path.join(
                base_path, col_config["folder_path"], row_config["subpath"]
            )

            print(f"Processing [{row_idx + 1},{col_idx + 1}]: {file_path}")

            # Check if we have cached image data
            if (row_idx, col_idx) in temp_images:
                events, _ = temp_images[(row_idx, col_idx)]

                # Determine if this column should be upscaled
                current_upscale_factor = 1
                if upscale_columns and col_idx in upscale_columns:
                    current_upscale_factor = upscale_factor
                    print(f"   🔍 Upscaling column {col_idx} by {upscale_factor}x")

                # For upscaled columns, use original size and let the function handle upscaling
                # For non-upscaled columns, use the unified size directly
                if current_upscale_factor > 1:
                    # Use original size (before upscaling) for upscaled columns
                    target_height = max_height // upscale_factor
                    target_width = max_width // upscale_factor
                else:
                    # Use unified size for non-upscaled columns
                    target_height = max_height
                    target_width = max_width

                # Create visualization with appropriate target size
                image = create_event_visualization(
                    events,
                    (target_height, target_width),
                    "both",
                    colors["positive"],
                    colors["negative"],
                    colors["background"],
                    use_density=use_density,
                    max_intensity=max_intensity,
                    upscale_factor=current_upscale_factor,
                    smooth_visualization=smooth_visualization,  # Use parameter
                    sigma=sigma,  # Use parameter
                    enhance_colors=enhance_colors,  # Use parameter
                    event_sample_ratio=event_sample_ratio,  # 🔥 NEW parameter
                    time_window=time_window,  # 🔥 NEW parameter
                    polarity_separation=polarity_separation,  # 🔥 NEW parameter
                )
            else:
                print(f"⚠️  File not found: {file_path}")
                # Create empty black image with unified size
                image = np.full(
                    (max_height, max_width, 3), colors["background"], dtype=np.float32
                )

            # Display image with fixed aspect ratio
            ax.imshow(image, aspect="equal", origin="upper", interpolation="nearest")

            # Set fixed axis limits to ensure consistent image sizes
            ax.set_xlim(0, max_width)
            ax.set_ylim(max_height, 0)  # Inverted for 'upper' origin

            # Add "File Not Found" text if needed
            if (row_idx, col_idx) not in temp_images:
                ax.text(
                    0.5,
                    0.5,
                    "File\nNot Found",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

            # Add magnified inset (only if enabled)
            if enable_magnification and bbox_configs and magnify_configs:
                bbox_config = bbox_configs[row_idx]
                magnify_config = magnify_configs[row_idx]

                # Use the same bbox coordinates for all images
                bbox_x = bbox_config["x"]
                bbox_y = bbox_config["y"]
                bbox_width = bbox_config["width"]
                bbox_height = bbox_config["height"]

                # Determine current upscale factor for this column
                current_upscale_factor = 1
                if upscale_columns and col_idx in upscale_columns:
                    current_upscale_factor = upscale_factor

                add_magnified_inset(
                    ax,
                    image,
                    bbox_x,
                    bbox_y,
                    bbox_width,
                    bbox_height,
                    magnify_position=magnify_config["position"],
                    magnify_scale=magnify_config["scale"],
                    magnify_color=colors["magnify"],
                    magnify_linewidth=2.0,
                    upscale_factor=current_upscale_factor,
                )

            # Set axis properties
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(colors["background"])

            # Add column labels (only for last row, at bottom)
            if show_column_labels and row_idx == n_rows - 1:
                # Choose text color based on background
                text_color = "black" if sum(colors["background"]) > 1.5 else "white"
                ax.set_xlabel(
                    col_config["label"],
                    fontsize=col_label_fontsize,
                    fontweight="bold",
                    color=text_color,
                    labelpad=col_label_pad,
                )

    # Adjust layout with customizable parameters FIRST
    # Don't use tight_layout as it interferes with manual spacing control
    if show_row_labels and show_column_labels:
        plt.subplots_adjust(
            left=left_margin, bottom=bottom_margin, wspace=wspace, hspace=hspace
        )
    elif show_row_labels:
        plt.subplots_adjust(left=left_margin, wspace=wspace, hspace=hspace)
    elif show_column_labels:
        plt.subplots_adjust(bottom=bottom_margin, wspace=wspace, hspace=hspace)
    else:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Add row labels AFTER layout adjustment - get actual subplot positions
    if show_row_labels:
        # Choose text color based on background
        text_color = "black" if sum(colors["background"]) > 1.5 else "white"
        for row_idx, row_config in enumerate(row_configs):
            # Get the actual position of the first subplot in this row
            ax_pos = axes[row_idx, 0].get_position()
            # Calculate the vertical center of this subplot
            row_center_y = ax_pos.y0 + (ax_pos.height / 2)

            # Add row label on the left
            fig.text(
                row_label_x,  # Customizable X position
                row_center_y,  # Actual subplot center Y position
                row_config["label"],
                fontsize=row_label_fontsize,  # Customizable font size
                fontweight="bold",
                color=text_color,
                ha="center",
                va="center",
                rotation=0,
            )

    # Save figure in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=colors["background"],
        edgecolor="none",
    )

    print(f"✅ Academic comparison grid saved to: {output_path}")
    plt.show()

    return fig, axes
