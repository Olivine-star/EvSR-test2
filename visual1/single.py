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
    negative_color=[0.0, 1.0, 0.0],
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
    magnify_color="lime",
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
    - magnify_color: Color of the bounding box
    """
    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Check if the image is too small for magnification
    if img_width < 20 or img_height < 20:
        print(
            f"⚠️  Image too small ({img_height}x{img_width}) for magnification. Skipping inset."
        )
        return

    # Ensure bbox is within image bounds
    bbox_x = max(0, min(bbox_x, img_width - bbox_width))
    bbox_y = max(0, min(bbox_y, img_height - bbox_height))
    bbox_width = min(bbox_width, img_width - bbox_x)
    bbox_height = min(bbox_height, img_height - bbox_y)

    # Extract the region to magnify
    magnified_region = image[
        bbox_y : bbox_y + bbox_height, bbox_x : bbox_x + bbox_width
    ]

    # Check if magnified region is valid
    if magnified_region.size == 0:
        print("⚠️  Magnified region is empty. Skipping inset.")
        return

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

    # Dynamic margin calculation based on image size
    # For small images, use smaller margins
    margin = max(2, min(8, img_width // 10, img_height // 10))

    # Determine inset position based on magnify_position
    # Note: In matplotlib, y=0 is at the top, y=1 is at the bottom for inset_axes
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
        raise ValueError(
            "magnify_position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'"
        )

    # Ensure inset position is within bounds
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

    # Display magnified region with clean border
    inset_ax.imshow(magnified_region, aspect="equal")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    # Add clean border around the inset with user-specified color
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(magnify_color)
        spine.set_linewidth(1.5)

    return inset_ax


def generate_single_visualization(
    event_path,
    bbox_x=10,
    bbox_y=10,
    bbox_width=15,
    bbox_height=15,
    magnify_position="bottom-right",
    magnify_scale=1.5,
    magnify_color="white",
    img_size=None,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],
    negative_color=[0.0, 1.0, 0.0],
    background_color=[0.0, 0.0, 0.0],
    output_filename=None,
    dpi=300,
    figsize=(8, 6),
    show_axes=False,
    show_title=False,
):
    """
    Generate event visualization image with optional axes and title

    Parameters:
    - event_path: Path to event .npy file
    - bbox_x, bbox_y: Top-left corner coordinates of magnification region
    - bbox_width, bbox_height: Dimensions of magnification region
    - magnify_position: Corner placement for inset
    - magnify_scale: Magnification factor
    - magnify_color: Color of the magnification bounding box
    - img_size: Image size tuple, if None will auto-determine from event data
    - polarity_filter: Event polarity filter
    - positive_color: RGB color for positive polarity events [R, G, B]
    - negative_color: RGB color for negative polarity events [R, G, B]
    - background_color: RGB color for background [R, G, B]
    - output_path: Output file path
    - dpi: Output resolution
    - figsize: Figure size (width, height)
    - show_axes: Whether to show coordinate axes
    - show_title: Whether to show title and labels

    Note: Can output pure image or image with axes and labels
    """

    # Check if file exists
    if not os.path.exists(event_path):
        print(f"❌ Event file does not exist: {event_path}")
        return

    # Generate output path in the same directory as input file
    if output_filename is None:
        # Extract filename without extension and add 2D suffix
        base_name = os.path.splitext(os.path.basename(event_path))[0]
        output_filename = f"{base_name}_2d.png"

    output_path = os.path.join(os.path.dirname(event_path), output_filename)

    # Load event data
    print("Loading event data...")
    events = np.load(event_path)

    # Determine image size
    if img_size is None:
        max_x = int(np.max(events[:, 1])) + 1
        max_y = int(np.max(events[:, 2])) + 1
        img_size = (max_y, max_x)
        print(f"Auto-determined image size: {img_size}")

    # Create visualization
    print("Creating visualization...")
    event_image, _ = create_event_visualization(
        events,
        img_size,
        polarity_filter,
        positive_color,
        negative_color,
        background_color,
    )

    # Create figure with optional axes and labels
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display the event image
    ax.imshow(event_image, aspect="equal", origin="upper")

    if show_axes:
        # Show coordinate axes with proper labels
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)

        # Set tick marks at reasonable intervals
        img_height, img_width = event_image.shape[:2]
        x_ticks = np.linspace(0, img_width - 1, min(10, img_width // 10 + 1), dtype=int)
        y_ticks = np.linspace(
            0, img_height - 1, min(10, img_height // 10 + 1), dtype=int
        )
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        if show_title:
            # Add informative title
            total_events = len(events)
            pos_events = np.sum(events[:, 3] > 0)
            neg_events = total_events - pos_events
            title = "Event Stream Visualization\n"
            title += (
                f"Total: {total_events} events (Pos: {pos_events}, Neg: {neg_events})\n"
            )
            title += f"Image size: {img_height}×{img_width} pixels"
            ax.set_title(title, fontsize=10, pad=20)

            # Create legend for polarity colors
            red_patch = patches.Patch(
                color=positive_color, label="Positive polarity (p > 0)"
            )
            blue_patch = patches.Patch(
                color=negative_color, label="Negative polarity (p ≤ 0)"
            )
            ax.legend(
                handles=[red_patch, blue_patch],
                loc="upper right",
                bbox_to_anchor=(1.15, 1),
            )
    else:
        # Remove all axes, labels, ticks, and borders for pure image
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # Add magnified inset if magnify_scale > 0
    if magnify_scale > 0 and bbox_width > 0 and bbox_height > 0:
        add_magnified_inset(
            ax,
            event_image,
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            magnify_position,
            magnify_scale,
            magnify_color,
        )

    # Adjust layout based on whether axes are shown
    if show_axes:
        plt.tight_layout()
    else:
        # Remove all margins and padding for pure image output
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure
    if show_axes:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    else:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor=background_color,
            edgecolor="none",
        )

    print(f"✅ Image saved to: {output_path}")

    # Display the figure
    plt.show()

    return fig, ax


# Example usage and configuration
if __name__ == "__main__":
    # Example file path - modify according to your data
    event_path = (
        r"C:\Users\steve\Project\EvSR-test2\visual1\test\cifar\cat\baseline.npy"
    )

    # User control parameters
    bbox_x = 0  # Top-left x coordinate of magnification region
    bbox_y = 0  # Top-left y coordinate of magnification region
    bbox_width = 0  # Width of magnification region
    bbox_height = 0  # Height of magnification region
    magnify_position = "bottom-right"  # Corner placement
    magnify_scale = 0  # Magnification factor
    magnify_color = "lime"  # Color of magnification border

    # Color control parameters (following memory: red for positive, blue for negative, white background)
    positive_color = [1.0, 0.0, 0.0]  # Red for positive events (p > 0)
    negative_color = [0.0, 0.0, 1.0]  # Blue for negative events (p ≤ 0)
    background_color = [1.0, 1.0, 1.0]  # White background

    # Generate visualization with axes and title
    generate_single_visualization(
        event_path=event_path,
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
        output_filename=None,  # Auto-generate based on input filename
        dpi=300,
        figsize=(10, 8),
        show_axes=False,
        show_title=False,
    )
