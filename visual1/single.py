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
    output_path="single_result.png",
    dpi=300,
    figsize=(8, 6),
):
    """
    Generate pure event visualization image without any text, axes, or labels

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

    Note: Outputs pure image without titles, axes, labels, or whitespace
    """

    # Check if file exists
    if not os.path.exists(event_path):
        print(f"❌ Event file does not exist: {event_path}")
        return

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

    # Create clean figure without any text or axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Remove all axes, labels, ticks, and borders
    ax.imshow(event_image, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Add magnified inset
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

    # Remove all margins and padding for pure image output
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save as pure image without any whitespace or borders
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=background_color,
        edgecolor="none",
    )
    print(f"✅ Pure image saved to: {output_path}")

    # Display the figure
    plt.show()

    return fig, ax


# Example usage and configuration
if __name__ == "__main__":
    # Example file path - modify according to your data
    event_path = r"C:\Users\steve\Project\EvSR-test2\visual1\test\a_0001_hr.npy"

    # User control parameters
    bbox_x = 100  # Top-left x coordinate of magnification region
    bbox_y = 100  # Top-left y coordinate of magnification region
    bbox_width = 30  # Width of magnification region
    bbox_height = 30  # Height of magnification region
    magnify_position = "bottom-right"  # Corner placement
    magnify_scale = 1.5  # Magnification factor
    magnify_color = "white"  # Color of magnification border

    # Color control parameters
    positive_color = [1.0, 0.0, 0.0]  # Red for positive events
    negative_color = [0.0, 1.0, 0.0]  # Green for negative events
    background_color = [0.0, 0.0, 0.0]  # Black background

    # Generate pure image visualization
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
        output_path="event_image.png",
        dpi=300,
        figsize=(8, 6),
    )
