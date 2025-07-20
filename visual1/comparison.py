import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os


def load_event_data(file_path):
    """
    Load event data from .npy file
    
    Args:
        file_path (str): Path to the .npy file containing event data
        
    Returns:
        numpy.ndarray: Event data with shape (N, 4) containing [timestamp, x, y, polarity]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Event data file not found: {file_path}")
    
    events = np.load(file_path)
    return events


def create_event_visualization(events, img_size=(256, 256), polarity_filter='both'):
    """
    Create event visualization from event data using the same logic as event_representation.ipynb
    
    Args:
        events (numpy.ndarray): Event data with shape (N, 4) containing [timestamp, x, y, polarity]
        img_size (tuple): Output image size (height, width)
        polarity_filter (str): 'both', 'positive', or 'negative'
        
    Returns:
        numpy.ndarray: Event visualization image
    """
    # Initialize pixel count array
    pixel_counts = np.zeros(img_size, dtype=int)
    
    # Count events per pixel based on polarity filter
    for event in events:
        ts, x, y, p = event
        
        # Apply polarity filter
        if polarity_filter == 'both' or (polarity_filter == 'positive' and p > 0) or (polarity_filter == 'negative' and p <= 0):
            # Use correct indexing: pixel_counts[int(y), int(x)] += 1
            pixel_counts[int(y), int(x)] += 1
    
    # Create RGB image with white background
    height, width = img_size
    rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Apply colors based on polarity
    if polarity_filter == 'both':
        # For combined visualization, use intensity-based coloring
        max_count = pixel_counts.max() if pixel_counts.max() > 0 else 1
        normalized_counts = pixel_counts / max_count
        
        # Create a purple-like visualization for combined events
        rgb_image[:, :, 0] = (255 * (1 - normalized_counts * 0.5)).astype(np.uint8)  # Red channel
        rgb_image[:, :, 1] = (255 * (1 - normalized_counts * 0.7)).astype(np.uint8)  # Green channel
        rgb_image[:, :, 2] = 255  # Blue channel stays high for purple effect
        
    elif polarity_filter == 'positive':
        # Red for positive polarity (p > 0)
        mask = pixel_counts > 0
        max_count = pixel_counts.max() if pixel_counts.max() > 0 else 1
        normalized_counts = pixel_counts / max_count
        
        rgb_image[mask, 0] = 255  # Full red
        rgb_image[mask, 1] = (255 * (1 - normalized_counts[mask])).astype(np.uint8)  # Reduce green
        rgb_image[mask, 2] = (255 * (1 - normalized_counts[mask])).astype(np.uint8)  # Reduce blue
        
    elif polarity_filter == 'negative':
        # Blue for negative polarity (p <= 0)
        mask = pixel_counts > 0
        max_count = pixel_counts.max() if pixel_counts.max() > 0 else 1
        normalized_counts = pixel_counts / max_count
        
        rgb_image[mask, 0] = (255 * (1 - normalized_counts[mask])).astype(np.uint8)  # Reduce red
        rgb_image[mask, 1] = (255 * (1 - normalized_counts[mask])).astype(np.uint8)  # Reduce green
        rgb_image[mask, 2] = 255  # Full blue
    
    return rgb_image


def create_combined_polarity_visualization(events, img_size=(256, 256)):
    """
    Create visualization showing both positive (red) and negative (blue) events
    
    Args:
        events (numpy.ndarray): Event data with shape (N, 4)
        img_size (tuple): Output image size (height, width)
        
    Returns:
        numpy.ndarray: RGB image with red for positive, blue for negative events
    """
    height, width = img_size
    rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Separate positive and negative events
    pos_counts = np.zeros(img_size, dtype=int)
    neg_counts = np.zeros(img_size, dtype=int)
    
    for event in events:
        ts, x, y, p = event
        if p > 0:
            pos_counts[int(y), int(x)] += 1
        else:
            neg_counts[int(y), int(x)] += 1
    
    # Apply red for positive events
    pos_mask = pos_counts > 0
    if pos_mask.any():
        max_pos = pos_counts.max()
        normalized_pos = pos_counts / max_pos
        rgb_image[pos_mask, 0] = 255  # Full red
        rgb_image[pos_mask, 1] = (255 * (1 - normalized_pos[pos_mask] * 0.8)).astype(np.uint8)
        rgb_image[pos_mask, 2] = (255 * (1 - normalized_pos[pos_mask] * 0.8)).astype(np.uint8)
    
    # Apply blue for negative events
    neg_mask = neg_counts > 0
    if neg_mask.any():
        max_neg = neg_counts.max()
        normalized_neg = neg_counts / max_neg
        # Only apply blue where there are no positive events to avoid mixing
        neg_only_mask = neg_mask & (~pos_mask)
        rgb_image[neg_only_mask, 0] = (255 * (1 - normalized_neg[neg_only_mask] * 0.8)).astype(np.uint8)
        rgb_image[neg_only_mask, 1] = (255 * (1 - normalized_neg[neg_only_mask] * 0.8)).astype(np.uint8)
        rgb_image[neg_only_mask, 2] = 255  # Full blue
    
    return rgb_image


def crop_and_magnify(image, bbox_x, bbox_y, bbox_width, bbox_height, magnify_position='top-right', magnify_scale=3):
    """
    Add a magnified crop region to an image
    
    Args:
        image (numpy.ndarray): Input image
        bbox_x, bbox_y (int): Top-left corner of bounding box
        bbox_width, bbox_height (int): Size of bounding box
        magnify_position (str): Position for magnified region ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        magnify_scale (int): Magnification factor
        
    Returns:
        numpy.ndarray: Image with magnified region added
    """
    result_image = image.copy()
    height, width = image.shape[:2]
    
    # Extract crop region
    crop_region = image[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
    
    # Calculate magnified size
    mag_height = bbox_height * magnify_scale
    mag_width = bbox_width * magnify_scale
    
    # Resize crop region (simple nearest neighbor for event data)
    magnified = np.repeat(np.repeat(crop_region, magnify_scale, axis=0), magnify_scale, axis=1)
    
    # Determine position for magnified region
    if magnify_position == 'top-left':
        start_y, start_x = 0, 0
    elif magnify_position == 'top-right':
        start_y, start_x = 0, width - mag_width
    elif magnify_position == 'bottom-left':
        start_y, start_x = height - mag_height, 0
    elif magnify_position == 'bottom-right':
        start_y, start_x = height - mag_height, width - mag_width
    else:
        raise ValueError("magnify_position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
    
    # Ensure magnified region fits within image bounds
    end_y = min(start_y + mag_height, height)
    end_x = min(start_x + mag_width, width)
    actual_mag_height = end_y - start_y
    actual_mag_width = end_x - start_x
    
    # Place magnified region
    result_image[start_y:end_y, start_x:end_x] = magnified[:actual_mag_height, :actual_mag_width]
    
    return result_image, (start_x, start_y, actual_mag_width, actual_mag_height)


def create_comparison_visualization(lr_path, hrgt_path, ours_path, 
                                  bbox_x=50, bbox_y=50, bbox_width=40, bbox_height=40,
                                  magnify_position='top-right', magnify_scale=3,
                                  img_size=(256, 256), save_path=None):
    """
    Create side-by-side comparison visualization with magnified crop regions
    
    Args:
        lr_path (str): Path to low resolution event data
        hrgt_path (str): Path to high resolution ground truth event data  
        ours_path (str): Path to our method's event data
        bbox_x, bbox_y (int): Top-left corner of bounding box for cropping
        bbox_width, bbox_height (int): Size of bounding box
        magnify_position (str): Position for magnified region
        magnify_scale (int): Magnification factor
        img_size (tuple): Image size for event visualization
        save_path (str): Optional path to save the comparison image
        
    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    # Load event data
    lr_events = load_event_data(lr_path)
    hrgt_events = load_event_data(hrgt_path)
    ours_events = load_event_data(ours_path)
    
    # Create event visualizations using combined polarity (red for positive, blue for negative)
    lr_image = create_combined_polarity_visualization(lr_events, img_size)
    hrgt_image = create_combined_polarity_visualization(hrgt_events, img_size)
    ours_image = create_combined_polarity_visualization(ours_events, img_size)
    
    # Add magnified crop regions
    lr_with_crop, lr_mag_bbox = crop_and_magnify(lr_image, bbox_x, bbox_y, bbox_width, bbox_height, magnify_position, magnify_scale)
    hrgt_with_crop, hrgt_mag_bbox = crop_and_magnify(hrgt_image, bbox_x, bbox_y, bbox_width, bbox_height, magnify_position, magnify_scale)
    ours_with_crop, ours_mag_bbox = crop_and_magnify(ours_image, bbox_x, bbox_y, bbox_width, bbox_height, magnify_position, magnify_scale)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Event Stream Comparison: LR vs HR-GT vs Ours', fontsize=16, fontweight='bold')
    
    # Display images
    titles = ['LR (Low Resolution)', 'HR-GT (Ground Truth)', 'Ours (Our Method)']
    images = [lr_with_crop, hrgt_with_crop, ours_with_crop]
    
    for i, (ax, image, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add bounding box for original crop region
        bbox_rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, 
                                    linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(bbox_rect)
        
        # Add bounding box for magnified region
        mag_x, mag_y, mag_w, mag_h = [lr_mag_bbox, hrgt_mag_bbox, ours_mag_bbox][i]
        mag_rect = patches.Rectangle((mag_x, mag_y), mag_w, mag_h,
                                   linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(mag_rect)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    return fig


# Example usage and main function
if __name__ == "__main__":
    # Example paths - update these with your actual file paths
    lr_path = r"C:\Users\steve\Project\EvSR-test2\visual\test\a_0001_train.npy"
    hrgt_path = r"C:\Users\steve\Project\EvSR-test2\visual\test\a_0001_train.npy"  # Replace with actual HR-GT path
    ours_path = r"C:\Users\steve\Project\EvSR-test2\visual\test\a_0001_train.npy"  # Replace with actual ours path
    
    # Create comparison visualization
    try:
        fig = create_comparison_visualization(
            lr_path=lr_path,
            hrgt_path=hrgt_path, 
            ours_path=ours_path,
            bbox_x=60,
            bbox_y=80,
            bbox_width=50,
            bbox_height=50,
            magnify_position='top-right',
            magnify_scale=3,
            img_size=(256, 256),
            save_path='comparison_result.png'
        )
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the file paths in the script to point to your actual event data files.")
