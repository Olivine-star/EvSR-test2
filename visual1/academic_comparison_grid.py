import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import os
import random

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]


def create_single_polarity_density_visualization(
    events, img_size, polarity_filter="positive", upscale_factor=1
):
    """
    Create density visualization following our_base对比.py logic EXACTLY
    Uses matplotlib colormap to generate pure image without axes or labels
    Supports 'both', 'positive', and 'negative' polarity filters
    Supports upscaling for LR images

    Parameters:
    - events: Event data array
    - img_size: Tuple (height, width) for output image size
    - polarity_filter: 'both', 'positive', or 'negative'
    - upscale_factor: Upscaling factor for LR images (default: 1)

    Returns:
    - RGB image array with density visualization (pure image, no axes)
    """
    # Apply upscaling if needed
    if upscale_factor > 1:
        # Scale up the target image size
        upscaled_size = (img_size[0] * upscale_factor, img_size[1] * upscale_factor)
        pixel_event_counts = np.zeros(upscaled_size, dtype=int)

        # Count events with upscaling
        for event in events:
            ts, x, y, p = event
            # Scale coordinates
            x_scaled = int(x * upscale_factor)
            y_scaled = int(y * upscale_factor)

            # Check bounds for upscaled image
            if 0 <= y_scaled < upscaled_size[0] and 0 <= x_scaled < upscaled_size[1]:
                # Apply polarity filter
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
                                pixel_event_counts[new_y, new_x] += 1

        # Update img_size to upscaled size
        img_size = upscaled_size
    else:
        # Original algorithm for non-upscaled images
        pixel_event_counts = np.zeros(img_size, dtype=int)

        # Count events for each pixel (following our_base对比.py logic exactly)
        for event in events:
            ts, x, y, p = event
            x_int, y_int = int(x), int(y)

            # Check bounds
            if 0 <= y_int < img_size[0] and 0 <= x_int < img_size[1]:
                # Apply polarity filter
                if (
                    polarity_filter == "both"
                    or (polarity_filter == "positive" and p > 0)
                    or (polarity_filter == "negative" and p <= 0)
                ):
                    pixel_event_counts[y_int, x_int] += 1

    # Use matplotlib colormap exactly like our_base对比.py
    if polarity_filter == "both":
        cmap_name = "Purples"
    elif polarity_filter == "positive":
        cmap_name = "Reds"
    elif polarity_filter == "negative":
        cmap_name = "Blues"
    else:
        cmap_name = "Purples"  # fallback

    # Get the colormap
    cmap = cm.get_cmap(cmap_name)

    # Normalize the counts to [0, 1] range for colormap
    if np.max(pixel_event_counts) > 0:
        normalized_counts = pixel_event_counts.astype(np.float32) / np.max(
            pixel_event_counts
        )
    else:
        normalized_counts = pixel_event_counts.astype(np.float32)

    # Apply colormap to get RGB image (exactly like plt.imshow does)
    rgb_image = cmap(normalized_counts)[:, :, :3]  # Remove alpha channel, keep RGB

    return rgb_image.astype(np.float32)


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
    colors=None,  # Color dictionary for gradient support
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
            colors,  # Pass colors for gradient support
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
    colors=None,  # Add colors parameter for gradient support
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
                # Mixed polarity: AGGRESSIVE blending with enhanced contrast
                total_count = pos_count + neg_count
                pos_ratio = pos_count / total_count
                neg_ratio = neg_count / total_count

                # Apply AGGRESSIVE polarity separation
                if polarity_separation != 1.0:
                    # Enhance the dominant polarity more aggressively
                    if pos_ratio > neg_ratio:
                        pos_ratio = min(pos_ratio * polarity_separation, 1.0)
                        neg_ratio = 1.0 - pos_ratio
                    else:
                        neg_ratio = min(neg_ratio * polarity_separation, 1.0)
                        pos_ratio = 1.0 - neg_ratio

                # AGGRESSIVE intensity calculation with power function
                raw_intensity = min(total_count / max(max_pos, max_neg), 1.0)
                intensity = (
                    raw_intensity**0.4
                ) * max_intensity  # More aggressive curve

                # AGGRESSIVE color blending with enhanced gradient colors
                if colors and "positive_max" in colors and "negative_max" in colors:
                    pos_color = (
                        np.array(positive_color) * (1 - intensity)
                        + np.array(colors["positive_max"]) * intensity
                    )
                    neg_color = (
                        np.array(negative_color) * (1 - intensity)
                        + np.array(colors["negative_max"]) * intensity
                    )
                    rgb_image[y, x] = pos_ratio * pos_color + neg_ratio * neg_color
                else:
                    # Fallback blending
                    rgb_image[y, x] = (
                        pos_ratio * np.array(positive_color) * intensity
                        + neg_ratio * np.array(negative_color) * intensity
                    )
            elif pos_count > 0:
                # Only positive events - AGGRESSIVE gradient from very light to pure red
                raw_intensity = min(pos_count / max_pos, 1.0)
                # Apply power function for more aggressive contrast
                intensity = raw_intensity**0.5  # Square root for more aggressive curve
                # Create AGGRESSIVE gradient: very light pink -> pure bright red
                base_color = np.array(positive_color)  # Very light pink base
                if colors and "positive_max" in colors:
                    max_color = np.array(colors["positive_max"])  # Pure bright red
                    # More aggressive interpolation with enhanced contrast
                    rgb_image[y, x] = (
                        base_color * (1 - intensity) + max_color * intensity
                    )
                else:
                    # Fallback with more aggressive scaling
                    rgb_image[y, x] = base_color * (0.1 + 0.9 * intensity)
            elif neg_count > 0:
                # Only negative events - AGGRESSIVE gradient from very light to pure blue
                raw_intensity = min(neg_count / max_neg, 1.0)
                # Apply power function for more aggressive contrast
                intensity = raw_intensity**0.5  # Square root for more aggressive curve
                # Create AGGRESSIVE gradient: very light blue -> pure bright blue
                base_color = np.array(negative_color)  # Very light blue base
                if colors and "negative_max" in colors:
                    max_color = np.array(colors["negative_max"])  # Pure bright blue
                    # More aggressive interpolation with enhanced contrast
                    rgb_image[y, x] = (
                        base_color * (1 - intensity) + max_color * intensity
                    )
                else:
                    # Fallback with more aggressive scaling
                    rgb_image[y, x] = base_color * (0.1 + 0.9 * intensity)

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
    show_bounding_box=True,  # Whether to show the bounding box frame
    show_magnified_inset=True,  # Whether to show the magnified inset
):
    """Add magnified inset with white border and/or bounding box
    bbox coordinates are in normalized coordinates (0-1) for extent=[0,1,0,1]

    Parameters:
    - show_bounding_box: If True, shows the bounding box frame
    - show_magnified_inset: If True, shows the magnified inset
    """
    img_height, img_width = image.shape[:2]

    # For image extraction, use original pixel coordinates directly
    # Don't use the flipped bbox_x, bbox_y which are for Rectangle display
    # Instead, extract from the original pixel coordinates passed to this function
    # But we need to get them from the calling function...
    # Actually, let's reconstruct from the normalized coordinates
    pixel_bbox_x = int(bbox_x * img_width)
    # For Y: Since we now use origin="lower", no need to flip Y coordinate
    pixel_bbox_y = int(bbox_y * img_height)  # Direct conversion, no flipping needed
    pixel_bbox_width = int(bbox_width * img_width)
    pixel_bbox_height = int(bbox_height * img_height)

    # Ensure pixel bbox is within bounds
    pixel_bbox_x = max(0, min(pixel_bbox_x, img_width - pixel_bbox_width))
    pixel_bbox_y = max(0, min(pixel_bbox_y, img_height - pixel_bbox_height))
    pixel_bbox_width = min(pixel_bbox_width, img_width - pixel_bbox_x)
    pixel_bbox_height = min(pixel_bbox_height, img_height - pixel_bbox_y)

    # Extract magnified region using pixel coordinates
    magnified_region = image[
        pixel_bbox_y : pixel_bbox_y + pixel_bbox_height,
        pixel_bbox_x : pixel_bbox_x + pixel_bbox_width,
    ]

    if magnified_region.size == 0:
        return

    # Add bounding box using data coordinate system (extent=[0,1,0,1]) - only if enabled
    if show_bounding_box:
        rect = Rectangle(
            (bbox_x, bbox_y),
            bbox_width,
            bbox_height,
            linewidth=magnify_linewidth,
            edgecolor=magnify_color,
            facecolor="none",
        )
        ax.add_patch(rect)

    # Create magnified inset - only if enabled
    if not show_magnified_inset:
        return None

    # Calculate inset position using pixel coordinates
    inset_width = int(pixel_bbox_width * magnify_scale)
    inset_height = int(pixel_bbox_height * magnify_scale)
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

    # Display magnified region with original image aspect ratio
    # Calculate the aspect ratio of the original image
    img_aspect_ratio = img_width / img_height  # 222/124 = 1.79 for NFS
    inset_ax.imshow(
        magnified_region,
        aspect=img_aspect_ratio,
        origin="lower",
        interpolation="nearest",
    )
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
    dpi=600,
    figsize_per_cell=(3, 3),
    show_row_labels=True,
    show_column_labels=True,
    show_bounding_boxes=True,  # Show bounding box frames (True/False)
    enable_magnification=True,  # Show magnified insets (True/False)
    transpose_layout=False,  # If True, swap rows and columns (transpose the grid)
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
    wspace=0,  # Width spacing between subplots
    hspace=0,  # Height spacing between subplots
    left_margin=0.05,  # Left margin for row labels
    bottom_margin=0.12,  # Bottom margin for column labels
    row_label_x=0.01,  # X position of row labels (0-1)
    row_label_fontsize=12,  # Font size for row labels
    col_label_fontsize=12,  # Font size for column labels
    col_label_y=0.02,  # Y distance of column labels from bottom of images
):
    """
    Generate academic comparison grid visualization

    Parameters:
    - base_path: Base directory path containing data folders
    - row_configs: List of dicts with row info: [{"label": "(1)", "subpath": "3/9.npy"}, ...]
    - column_configs: List of dicts with column info: [{"label": "LR", "folder_path": "light-p-learn/HRPre"}, ...]
    - transpose_layout: If True, swap rows and columns (transpose the grid)
    """

    # Store original configs for potential transpose
    original_row_configs = row_configs
    original_column_configs = column_configs

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
    # Handle transpose: swap dimensions
    if transpose_layout:
        n_rows = len(original_column_configs)
        n_cols = len(original_row_configs)
        # Create transposed configs with correct dimensions
        row_configs = original_column_configs
        column_configs = original_row_configs
    else:
        n_rows = len(row_configs)
        n_cols = len(column_configs)
    figsize = (figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)

    # Create figure using subplots (like DRAW_NMNIST.py)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
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
            # Handle path construction for both normal and transposed layouts
            if transpose_layout:
                # In transpose: use original indices to access correct data
                # row_idx maps to original col_idx, col_idx maps to original row_idx
                original_row_config = original_row_configs[col_idx]
                original_col_config = original_column_configs[row_idx]
                file_path = os.path.join(
                    base_path,
                    original_col_config["folder_path"],
                    original_row_config["subpath"],
                )
            else:
                # Normal layout: base_path + column_folder_path + row_subpath
                file_path = os.path.join(
                    base_path, col_config["folder_path"], row_config["subpath"]
                )
            if os.path.exists(file_path):
                events = np.load(file_path)
                max_x = int(np.max(events[:, 1])) + 1
                max_y = int(np.max(events[:, 2])) + 1

                # Apply upscaling factor if this column/row should be upscaled
                should_upscale = False
                if upscale_columns:
                    if transpose_layout:
                        # In transpose mode, upscale_columns refers to original columns (now rows)
                        should_upscale = row_idx in upscale_columns
                    else:
                        # Normal mode, upscale_columns refers to columns
                        should_upscale = col_idx in upscale_columns

                if should_upscale:
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

            # Construct file path: handle both normal and transposed layouts
            if transpose_layout:
                # In transpose: use original indices to access correct data
                original_row_config = original_row_configs[col_idx]
                original_col_config = original_column_configs[row_idx]
                file_path = os.path.join(
                    base_path,
                    original_col_config["folder_path"],
                    original_row_config["subpath"],
                )
            else:
                # Normal layout: base_path + column_folder_path + row_subpath
                file_path = os.path.join(
                    base_path, col_config["folder_path"], row_config["subpath"]
                )

            print(f"Processing [{row_idx + 1},{col_idx + 1}]: {file_path}")

            # Check if we have cached image data
            if (row_idx, col_idx) in temp_images:
                events, _ = temp_images[(row_idx, col_idx)]

                # Determine if this column/row should be upscaled
                current_upscale_factor = 1
                should_upscale = False
                if upscale_columns:
                    if transpose_layout:
                        # In transpose mode, upscale_columns refers to original columns (now rows)
                        should_upscale = row_idx in upscale_columns
                    else:
                        # Normal mode, upscale_columns refers to columns
                        should_upscale = col_idx in upscale_columns

                if should_upscale:
                    current_upscale_factor = upscale_factor
                    if transpose_layout:
                        print(f"   🔍 Upscaling row {row_idx} by {upscale_factor}x")
                    else:
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

                # Check if this row has polarity parameter specified
                # In transpose mode, use original row config for polarity
                current_row_config = (
                    original_row_configs[col_idx] if transpose_layout else row_config
                )
                if "polarity" in current_row_config:
                    # Use the new density visualization method (following our_base对比.py)
                    row_polarity = current_row_config[
                        "polarity"
                    ]  # Get the specified polarity
                    image = create_single_polarity_density_visualization(
                        events,
                        (target_height, target_width),
                        polarity_filter=row_polarity,
                        upscale_factor=current_upscale_factor,  # Pass upscale factor
                    )
                else:
                    # Use default original visualization method (no polarity parameter specified)
                    image = create_event_visualization(
                        events,
                        (target_height, target_width),
                        "both",  # Default to both polarities
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
                        colors=colors,  # 🌈 Pass colors for gradient support
                    )
            else:
                print(f"⚠️  File not found: {file_path}")
                # Create empty black image with unified size
                image = np.full(
                    (max_height, max_width, 3), colors["background"], dtype=np.float32
                )

            # Display image using DRAW_NMNIST.py method with proper aspect ratio
            ax.imshow(
                image,
                aspect="auto",  # Allow automatic aspect ratio based on image dimensions
                origin="lower",  # Use lower origin to match event camera coordinate system
                interpolation="nearest",
                extent=[0, 1, 0, 1],
            )

            # Set up table-like appearance with borders
            ax.set_xticks([])
            ax.set_yticks([])

            # Keep all borders visible for table appearance
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.5)

            # Hide axis labels for inner cells, keep for edge cells with labels
            if not (show_column_labels and row_idx == n_rows - 1):
                ax.set_xlabel("")  # Hide x-axis label for non-bottom cells
            if not (show_row_labels and col_idx == 0):
                ax.set_ylabel("")  # Hide y-axis label for non-left cells

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

            # Add bounding box and/or magnified inset (if either is enabled)
            if (
                (show_bounding_boxes or enable_magnification)
                and bbox_configs
                and magnify_configs
            ):
                # Determine config index based on mode
                if transpose_layout:
                    # In transpose mode: current col_idx maps to original row_idx
                    # But we need to ensure it's within bounds
                    config_idx = col_idx if col_idx < len(bbox_configs) else 0
                else:
                    # Normal mode: use row_idx
                    config_idx = row_idx

                # Safety check for bounds
                if config_idx < len(bbox_configs) and config_idx < len(magnify_configs):
                    bbox_config = bbox_configs[config_idx]
                    magnify_config = magnify_configs[config_idx]

                    # Convert pixel coordinates to extent=[0,1,0,1] coordinates
                    # extent=[0,1,0,1] means (0,0) is bottom-left, (1,1) is top-right
                    # But origin="upper" flips the image display
                    # For Rectangle: need to flip Y coordinate
                    img_height, img_width = image.shape[:2]
                    bbox_x = bbox_config["x"] / img_width
                    bbox_y = (
                        1.0 - (bbox_config["y"] + bbox_config["height"]) / img_height
                    )
                    bbox_width = bbox_config["width"] / img_width
                    bbox_height = bbox_config["height"] / img_height

                    # Determine current upscale factor for this column/row
                    current_upscale_factor = 1
                    should_upscale = False
                    if upscale_columns:
                        if transpose_layout:
                            # In transpose mode, upscale_columns refers to original columns (now rows)
                            should_upscale = row_idx in upscale_columns
                        else:
                            # Normal mode, upscale_columns refers to columns
                            should_upscale = col_idx in upscale_columns

                    if should_upscale:
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
                        show_bounding_box=show_bounding_boxes,  # Pass the parameter
                        show_magnified_inset=enable_magnification,  # Pass the parameter
                    )

            # Set axis properties
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(colors["background"])

            # Column labels will be added separately after layout adjustment

    # Apply spacing adjustment using the provided parameters
    plt.subplots_adjust(
        wspace=wspace,  # Use provided horizontal spacing
        hspace=hspace,  # Use provided vertical spacing
        left=left_margin,  # Use provided left margin
        bottom=bottom_margin,  # Use provided bottom margin
        right=0.98,
        top=0.98,
    )

    # Add row labels AFTER layout adjustment - get actual subplot positions
    if show_row_labels:
        # Choose text color based on background
        text_color = "black" if sum(colors["background"]) > 1.5 else "white"
        for row_idx, row_config in enumerate(row_configs):
            # Get the actual position of the first subplot in this row
            ax_pos = axes[row_idx, 0].get_position()
            # Calculate the vertical center of this subplot
            row_center_y = ax_pos.y0 + (ax_pos.height / 2)

            # Add row label on the left, always outside the plot area
            # Use row_label_x parameter to control distance from edge
            label_x = ax_pos.x0 - (
                row_label_x * 10
            )  # Scale row_label_x for better control

            # Determine rotation based on label content
            label_text = row_config["label"]
            # Check if label contains letters (not just numbers, parentheses, spaces)
            has_letters = any(c.isalpha() for c in label_text)
            rotation = 90 if has_letters else 0  # Rotate 90 degrees if has letters

            fig.text(
                label_x,
                row_center_y,  # Actual subplot center Y position
                label_text,
                fontsize=row_label_fontsize,  # Customizable font size
                fontweight="bold",
                color=text_color,
                ha="right",  # Right-align so text doesn't overlap with images
                va="center",
                rotation=rotation,  # Dynamic rotation based on content
            )

    # Add column labels AFTER layout adjustment - place them outside the plot area
    if show_column_labels:
        # Choose text color based on background
        text_color = "black" if sum(colors["background"]) > 1.5 else "white"
        for col_idx, col_config in enumerate(column_configs):
            # Get the actual position of the bottom subplot in this column
            ax_pos = axes[n_rows - 1, col_idx].get_position()

            # Calculate label position (below the subplot, outside plot area)
            label_x = ax_pos.x0 + ax_pos.width / 2  # Center horizontally
            label_y = ax_pos.y0 - col_label_y  # Use col_label_y parameter for distance

            # Add column label as figure text (outside the subplot area)
            fig.text(
                label_x,
                label_y,
                col_config["label"],
                fontsize=col_label_fontsize,
                fontweight="bold",
                color=text_color,
                ha="center",
                va="top",  # Top-align so text doesn't overlap with images
                rotation=0,
            )

    # Save figure in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    # Check if output is PDF and adjust settings accordingly
    if output_filename.lower().endswith(".pdf"):
        plt.savefig(
            output_path,
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=colors["background"],
            edgecolor="none",
        )
    else:
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
