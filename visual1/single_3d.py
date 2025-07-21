import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def generate_3d_point_cloud_visualization(
    event_path,
    time_scale=1.0,
    point_size=6.0,
    alpha=0.6,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],
    negative_color=[0.0, 0.0, 1.0],
    background_color=[1.0, 1.0, 1.0],
    elevation=30,
    azimuth=60,
    output_path="event_3d_pointcloud.png",
    dpi=300,
    figsize=(10, 8),
    max_events=20000,
    show_axes=False,
):
    """
    Generate 3D point cloud visualization of event stream data similar to research paper style

    Parameters:
    - event_path: Path to event .npy file
    - time_scale: Scale factor for time axis (smaller = more compressed in time)
    - point_size: Size of individual points
    - alpha: Transparency of points (0.0-1.0)
    - polarity_filter: 'both', 'positive', or 'negative'
    - positive_color: RGB color for positive polarity events [R, G, B]
    - negative_color: RGB color for negative polarity events [R, G, B]
    - background_color: RGB color for background [R, G, B]
    - elevation: Viewing elevation angle (degrees)
    - azimuth: Viewing azimuth angle (degrees)
    - output_path: Output file path
    - dpi: Output resolution
    - figsize: Figure size (width, height)
    - max_events: Maximum number of events to display (for performance)
    - show_axes: Whether to show coordinate axes

    Note: Creates 3D point cloud where (x, y, time) represent spatial-temporal coordinates
    """

    # Check if file exists
    if not os.path.exists(event_path):
        print(f"❌ Event file does not exist: {event_path}")
        return

    # Load event data
    print("Loading event data...")
    events = np.load(event_path)
    print(f"Total events: {len(events)}")

    # Subsample events if too many for performance
    if len(events) > max_events:
        indices = np.random.choice(len(events), max_events, replace=False)
        events = events[indices]
        print(f"Subsampled to {max_events} events for performance")

    # Extract coordinates and polarities
    timestamps = events[:, 0]
    x_coords = events[:, 1]
    y_coords = events[:, 2]
    polarities = events[:, 3]

    print("Original data ranges:")
    print(f"  Time: {timestamps.min():.2f} to {timestamps.max():.2f}")
    print(f"  X: {x_coords.min():.0f} to {x_coords.max():.0f}")
    print(f"  Y: {y_coords.min():.0f} to {y_coords.max():.0f}")
    print(f"  Polarity: {np.unique(polarities)}")

    # For 3D visualization, use timestamps as discrete layers
    # Keep original integer timestamps for better layer separation
    print(f"Unique timestamps: {len(np.unique(timestamps))}")
    print(f"Time layers: {np.unique(timestamps)[:10]}...")

    # Apply minimal scaling to separate time layers visually
    timestamps = timestamps * time_scale

    print(f"After time scaling (scale={time_scale}):")
    print(f"  Time: {timestamps.min():.2f} to {timestamps.max():.2f}")

    # Filter events based on polarity
    if polarity_filter == "positive":
        mask = polarities > 0
    elif polarity_filter == "negative":
        mask = polarities <= 0
    else:  # both
        mask = np.ones(len(events), dtype=bool)

    # Apply filter
    timestamps = timestamps[mask]
    x_coords = x_coords[mask]
    y_coords = y_coords[mask]
    polarities = polarities[mask]

    print(f"Filtered events: {len(timestamps)}")

    # Create 3D figure
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background_color)

    # Separate positive and negative events
    pos_mask = polarities > 0
    neg_mask = polarities <= 0

    # Plot positive events (red dots)
    if np.any(pos_mask) and (polarity_filter in ["both", "positive"]):
        ax.scatter(
            x_coords[pos_mask],
            y_coords[pos_mask],
            timestamps[pos_mask],
            c=positive_color,
            s=point_size,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )

    # Plot negative events (blue dots)
    if np.any(neg_mask) and (polarity_filter in ["both", "negative"]):
        ax.scatter(
            x_coords[neg_mask],
            y_coords[neg_mask],
            timestamps[neg_mask],
            c=negative_color,
            s=point_size,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    if not show_axes:
        # Remove axes, labels, and grid for clean output
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        # Remove axis labels and title
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # Make axes invisible
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)

        # Remove the panes (background planes)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")
    else:
        # Show axes with labels (research paper style)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_zlabel("t", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Remove margins for clean output
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save as pure image
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=background_color,
        edgecolor="none",
        transparent=False,
    )
    print(f"✅ 3D point cloud visualization saved to: {output_path}")

    # Display the figure
    plt.show()

    return fig, ax


def generate_3d_density_visualization(
    event_path,
    time_bins=50,
    spatial_bins=32,
    time_scale=1.0,
    alpha=0.7,
    polarity_filter="both",
    positive_color=[1.0, 0.0, 0.0],
    negative_color=[0.0, 1.0, 0.0],
    background_color=[0.0, 0.0, 0.0],
    elevation=20,
    azimuth=45,
    output_path="event_3d_density.png",
    dpi=300,
    figsize=(10, 8),
):
    """
    Generate 3D density visualization using voxels

    Parameters similar to point cloud version, plus:
    - time_bins: Number of time bins for voxelization
    - spatial_bins: Number of spatial bins per dimension
    """

    # Check if file exists
    if not os.path.exists(event_path):
        print(f"❌ Event file does not exist: {event_path}")
        return

    # Load event data
    print("Loading event data...")
    events = np.load(event_path)

    # Extract coordinates and polarities
    timestamps = events[:, 0]
    x_coords = events[:, 1].astype(int)
    y_coords = events[:, 2].astype(int)
    polarities = events[:, 3]

    # Normalize timestamps
    timestamps = timestamps - timestamps.min()
    timestamps = timestamps * time_scale

    # Get data ranges
    x_max, y_max = x_coords.max(), y_coords.max()
    t_max = timestamps.max()

    # Create 3D histogram (voxel grid)
    pos_mask = polarities > 0
    neg_mask = polarities <= 0

    # Create voxel grids
    edges = None
    if polarity_filter in ["both", "positive"] and np.any(pos_mask):
        pos_hist, edges = np.histogramdd(
            np.column_stack(
                [x_coords[pos_mask], y_coords[pos_mask], timestamps[pos_mask]]
            ),
            bins=[spatial_bins, spatial_bins, time_bins],
            range=[[0, x_max], [0, y_max], [0, t_max]],
        )
    else:
        pos_hist = np.zeros((spatial_bins, spatial_bins, time_bins))

    if polarity_filter in ["both", "negative"] and np.any(neg_mask):
        neg_hist, edges_neg = np.histogramdd(
            np.column_stack(
                [x_coords[neg_mask], y_coords[neg_mask], timestamps[neg_mask]]
            ),
            bins=[spatial_bins, spatial_bins, time_bins],
            range=[[0, x_max], [0, y_max], [0, t_max]],
        )
        if edges is None:
            edges = edges_neg
    else:
        neg_hist = np.zeros((spatial_bins, spatial_bins, time_bins))

    # If no edges were created, create default ones
    if edges is None:
        edges = [
            np.linspace(0, x_max, spatial_bins + 1),
            np.linspace(0, y_max, spatial_bins + 1),
            np.linspace(0, t_max, time_bins + 1),
        ]

    # Create 3D figure
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background_color)

    # Create coordinate grids
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    t_centers = (edges[2][:-1] + edges[2][1:]) / 2

    # Plot positive voxels
    pos_indices = np.where(pos_hist > 0)
    if len(pos_indices[0]) > 0:
        ax.scatter(
            x_centers[pos_indices[0]],
            y_centers[pos_indices[1]],
            t_centers[pos_indices[2]],
            c=[positive_color],
            s=pos_hist[pos_indices] * 10,  # Size proportional to density
            alpha=alpha,
        )

    # Plot negative voxels
    neg_indices = np.where(neg_hist > 0)
    if len(neg_indices[0]) > 0:
        ax.scatter(
            x_centers[neg_indices[0]],
            y_centers[neg_indices[1]],
            t_centers[neg_indices[2]],
            c=[negative_color],
            s=neg_hist[neg_indices] * 10,  # Size proportional to density
            alpha=alpha,
        )

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Clean up axes (same as point cloud version)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save image
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=background_color,
        edgecolor="none",
        transparent=False,
    )
    print(f"✅ 3D density visualization saved to: {output_path}")

    # Display the figure
    plt.show()

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Example file path
    event_path = r"C:\Users\steve\Project\EvSR-test2\visual1\test\a_0001_hr.npy"

    # 3D Point Cloud Parameters (research paper style)
    time_scale = 1.0  # Keep time layers distinct (0-97 time frames)
    point_size = 6.0  # Smaller points to see shape better
    alpha = 0.6  # Lower opacity to see through layers
    elevation = 30  # Better viewing angle for thumb shape
    azimuth = 60  # Rotate to see 3D structure

    # Color control (research paper style)
    positive_color = [1.0, 0.0, 0.0]  # Red for positive events
    negative_color = [0.0, 0.0, 1.0]  # Blue for negative events
    background_color = [1.0, 1.0, 1.0]  # White background

    # Generate 3D point cloud visualization with axes (research style)
    generate_3d_point_cloud_visualization(
        event_path=event_path,
        time_scale=time_scale,
        point_size=point_size,
        alpha=alpha,
        positive_color=positive_color,
        negative_color=negative_color,
        background_color=background_color,
        elevation=elevation,
        azimuth=azimuth,
        polarity_filter="both",
        output_path="event_3d_pointcloud_axes.png",
        dpi=300,
        max_events=15000,  # Limit for performance
        show_axes=True,  # Show axes like research paper
    )
