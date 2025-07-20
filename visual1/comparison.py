import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os


def create_event_visualization(
    events,
    img_size,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],
    negative_color=[0.0, 0.8, 1.0],
    background_color=[0.0, 0.0, 0.0],
):
    """
    Create event visualization following the same logic as event_representation.ipynb

    Parameters:
    - events: Event data containing [timestamp, x, y, polarity]
    - img_size: Output image size (height, width)
    - polarity_filter: 'both', 'positive', or 'negative'
    - positive_color: RGB color for positive polarity events [R, G, B]
    - negative_color: RGB color for negative polarity events [R, G, B]
    - background_color: RGB color for background [R, G, B]

    Returns:
    - RGB image array for visualization
    """
    # Initialize pixel counts
    pixel_counts = np.zeros(img_size, dtype=int)

    # Process events with correct pixel indexing: pixel_counts[int(y), int(x)] += 1
    for event in events:
        ts, x, y, p = event
        x_int, y_int = int(x), int(y)

        # Check bounds to prevent index errors
        if 0 <= y_int < img_size[0] and 0 <= x_int < img_size[1]:
            # Apply polarity filter
            if (
                polarity_filter == "both"
                or (polarity_filter == "positive" and p > 0)
                or (polarity_filter == "negative" and p <= 0)
            ):
                pixel_counts[y_int, x_int] += 1

    # Create RGB visualization with user-controllable colors
    rgb_image = np.full(
        (img_size[0], img_size[1], 3), background_color, dtype=np.float32
    )

    # Process events for color assignment
    for event in events:
        ts, x, y, p = event
        x_int, y_int = int(x), int(y)

        # Check bounds to prevent index errors
        if 0 <= y_int < img_size[0] and 0 <= x_int < img_size[1]:
            if polarity_filter == "both":
                if p > 0:  # Positive polarity
                    rgb_image[y_int, x_int] = positive_color
                else:  # Negative polarity
                    rgb_image[y_int, x_int] = negative_color
            elif polarity_filter == "positive" and p > 0:
                rgb_image[y_int, x_int] = positive_color
            elif polarity_filter == "negative" and p <= 0:
                rgb_image[y_int, x_int] = negative_color

    return rgb_image, pixel_counts


def add_magnified_inset(
    ax,
    image,
    bbox_x,
    bbox_y,
    bbox_width,
    bbox_height,
    magnify_position="top-right",
    magnify_scale=1.5,
    magnify_color="black",
):
    """
    Add magnified inset to an axis with clean bounding box

    Parameters:
    - ax: Matplotlib axis object
    - image: Image array to magnify
    - bbox_x, bbox_y: Top-left corner of region to magnify
    - bbox_width, bbox_height: Dimensions of region to magnify
    - magnify_position: Corner placement ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    - magnify_scale: Magnification factor
    - magnify_color: Color of magnification border
    """
    # Extract the region to magnify
    magnified_region = image[
        bbox_y : bbox_y + bbox_height, bbox_x : bbox_x + bbox_width
    ]

    # Add clean bounding box around the original region
    rect = Rectangle(
        (bbox_x, bbox_y),
        bbox_width,
        bbox_height,
        linewidth=1.5,
        edgecolor=magnify_color,
        facecolor="none",
    )
    ax.add_patch(rect)

    # Calculate inset position and size
    img_height, img_width = image.shape[:2]
    inset_width = int(bbox_width * magnify_scale)
    inset_height = int(bbox_height * magnify_scale)

    # Determine inset position based on magnify_position
    # Note: In matplotlib, y=0 is at the top, y=1 is at the bottom for inset_axes
    margin = 8  # Margin from edges
    if magnify_position == "top-right":
        inset_x = img_width - inset_width - margin
        inset_y = (
            img_height - inset_height - margin
        )  # Fixed: top means bottom in inset coordinates
    elif magnify_position == "top-left":
        inset_x = margin
        inset_y = (
            img_height - inset_height - margin
        )  # Fixed: top means bottom in inset coordinates
    elif magnify_position == "bottom-right":
        inset_x = img_width - inset_width - margin
        inset_y = margin  # Fixed: bottom means top in inset coordinates
    elif magnify_position == "bottom-left":
        inset_x = margin
        inset_y = margin  # Fixed: bottom means top in inset coordinates
    else:
        raise ValueError(
            "magnify_position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'"
        )

    # Create inset axes
    inset_ax = ax.inset_axes(
        [
            inset_x / img_width,
            inset_y / img_height,
            inset_width / img_width,
            inset_height / img_height,
        ]
    )

    # Display magnified region with clean border
    inset_ax.imshow(magnified_region, aspect="equal")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    # Add clean border around the inset
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(magnify_color)
        spine.set_linewidth(1.5)

    return inset_ax


def generate_comparison_visualization(
    lr_path,
    hrgt_path,
    ours_path,
    bbox_x=10,
    bbox_y=10,
    bbox_width=15,
    bbox_height=15,
    magnify_position="top-right",
    magnify_scale=1.5,
    magnify_color="black",
    img_size=None,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],
    negative_color=[0.0, 0.8, 1.0],
    background_color=[0.0, 0.0, 0.0],
    output_path="comparison_result.png",
    dpi=300,
):
    """
    Generate academic paper-quality comparison visualization with unified scaling

    Parameters:
    - lr_path: Path to Low Resolution .npy file
    - hrgt_path: Path to High Resolution Ground Truth .npy file
    - ours_path: Path to Our Method .npy file
    - bbox_x, bbox_y: Top-left corner coordinates of magnification region
    - bbox_width, bbox_height: Dimensions of magnification region
    - magnify_position: Corner placement for inset (fixed coordinates)
    - magnify_scale: Magnification factor
    - magnify_color: Color of magnification border
    - img_size: Image size tuple, if None will use unified size from all datasets
    - polarity_filter: Event polarity filter
    - positive_color: RGB color for positive polarity events [R, G, B]
    - negative_color: RGB color for negative polarity events [R, G, B]
    - background_color: RGB color for background [R, G, B]
    - output_path: Output file path
    - dpi: Output resolution

    Note: All images will be displayed at the same scale for proper comparison
    """

    # Check if files exist
    for path, name in [(lr_path, "LR"), (hrgt_path, "HR-GT"), (ours_path, "Ours")]:
        if not os.path.exists(path):
            print(f"❌ {name} file does not exist: {path}")
            return

    # Load event data
    print("Loading event data...")
    lr_events = np.load(lr_path)
    hrgt_events = np.load(hrgt_path)
    ours_events = np.load(ours_path)

    # Determine image sizes for proper comparison
    if img_size is None:
        # Get individual sizes first
        lr_max_x = int(np.max(lr_events[:, 1])) + 1
        lr_max_y = int(np.max(lr_events[:, 2])) + 1
        hr_max_x = int(np.max(hrgt_events[:, 1])) + 1
        hr_max_y = int(np.max(hrgt_events[:, 2])) + 1
        ours_max_x = int(np.max(ours_events[:, 1])) + 1
        ours_max_y = int(np.max(ours_events[:, 2])) + 1

        print(f"LR size: ({lr_max_y}, {lr_max_x})")
        print(f"HR-GT size: ({hr_max_y}, {hr_max_x})")
        print(f"Ours size: ({ours_max_y}, {ours_max_x})")

        # Use HR size as reference for display
        display_size = (hr_max_y, hr_max_x)
        lr_native_size = (lr_max_y, lr_max_x)
    else:
        display_size = img_size
        # Assume LR is half the size of display for typical super-resolution
        lr_native_size = (img_size[0] // 2, img_size[1] // 2)

    # Create visualizations
    print("Creating visualizations...")
    # LR: create at native resolution, then upscale for display
    lr_image_native, _ = create_event_visualization(
        lr_events,
        lr_native_size,
        polarity_filter,
        positive_color,
        negative_color,
        background_color,
    )

    # Upscale LR image using numpy repeat for pixel-perfect scaling
    scale_y = display_size[0] // lr_native_size[0]
    scale_x = display_size[1] // lr_native_size[1]
    lr_image = np.repeat(np.repeat(lr_image_native, scale_y, axis=0), scale_x, axis=1)

    # If sizes don't match exactly, pad or crop to match display_size
    if lr_image.shape[:2] != display_size:
        lr_image_resized = np.full(
            (display_size[0], display_size[1], 3), background_color, dtype=np.float32
        )
        min_h = min(lr_image.shape[0], display_size[0])
        min_w = min(lr_image.shape[1], display_size[1])
        lr_image_resized[:min_h, :min_w] = lr_image[:min_h, :min_w]
        lr_image = lr_image_resized

    # HR-GT and Ours: create at display resolution
    hrgt_image, _ = create_event_visualization(
        hrgt_events,
        display_size,
        polarity_filter,
        positive_color,
        negative_color,
        background_color,
    )
    ours_image, _ = create_event_visualization(
        ours_events,
        display_size,
        polarity_filter,
        positive_color,
        negative_color,
        background_color,
    )

    # Create publication-ready figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Event Stream Super-Resolution Comparison", fontsize=16, fontweight="bold"
    )

    # Plot LR
    axes[0].imshow(lr_image, aspect="equal")
    axes[0].set_title("LR", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("x [pixels]", fontsize=12)
    axes[0].set_ylabel("y [pixels]", fontsize=12)
    add_magnified_inset(
        axes[0],
        lr_image,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        magnify_position,
        magnify_scale,
        magnify_color,
    )

    # Plot HR-GT
    axes[1].imshow(hrgt_image, aspect="equal")
    axes[1].set_title("HR-GT", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("x [pixels]", fontsize=12)
    axes[1].set_ylabel("y [pixels]", fontsize=12)
    add_magnified_inset(
        axes[1],
        hrgt_image,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        magnify_position,
        magnify_scale,
        magnify_color,
    )

    # Plot Ours
    axes[2].imshow(ours_image, aspect="equal")
    axes[2].set_title("Ours", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("x [pixels]", fontsize=12)
    axes[2].set_ylabel("y [pixels]", fontsize=12)
    add_magnified_inset(
        axes[2],
        ours_image,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        magnify_position,
        magnify_scale,
        magnify_color,
    )

    # Adjust layout for publication quality
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save high-resolution figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"✅ Comparison visualization saved to: {output_path}")

    # Display the figure
    plt.show()

    return fig, axes


# Example usage and configuration
if __name__ == "__main__":
    # Example file paths - modify these according to your data structure
    lr_path = r"C:\Users\steve\Project\EvSR-test2\visual1\test\a_0001_lr.npy"
    hrgt_path = r"C:\Users\steve\Project\EvSR-test2\visual1\test\a_0001_hr.npy"
    ours_path = r"C:\Users\steve\Project\EvSR-test2\visual1\test\a_0001_train.npy"

    # User control parameters
    bbox_x = 100  # Top-left x coordinate of magnification region
    bbox_y = 100  # Top-left y coordinate of magnification region
    bbox_width = 30  # Width of magnification region
    bbox_height = 30  # Height of magnification region
    magnify_position = "bottom-right"  # Corner placement: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    magnify_scale = 1.5  # Magnification factor

    # Color control parameters
    magnify_color = "White"  # Color of magnification border
    positive_color = [1.0, 0.0, 0.0]  # Red for positive events
    negative_color = [0.0, 1, 0]  # Bright green for negative events
    background_color = [0.0, 0.0, 0.0]  # Black background

    # Generate comparison
    generate_comparison_visualization(
        lr_path=lr_path,
        hrgt_path=hrgt_path,
        ours_path=ours_path,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        magnify_position=magnify_position,
        magnify_scale=magnify_scale,
        magnify_color=magnify_color,
        positive_color=positive_color,
        negative_color=negative_color,
        background_color=background_color,
        polarity_filter="both",
        output_path="comparison.png",
        dpi=300,
    )
