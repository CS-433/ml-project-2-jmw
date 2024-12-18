import numpy as np
import matplotlib.image as mpimg
import os 


def create_channel(points, shape, radius=2):
    """
    Creates an image channel of the given shape where circles of a given radius
    around point1 and point2 are set to 255, and the rest are zeros.

    Args:
        list of points (x, y) or [x, y]
        shape (tuple): Shape of the output channel (height, width).
        radius (int): Radius of the circles to draw around the points.

    Returns:
        numpy.ndarray: A 2D array representing the channel.
    """
    # Create a blank channel
    channel = np.zeros(shape, dtype=np.uint8)

    # Helper function to draw a circle around a point
    def draw_circle(center, radius):
        cx, cy = center
        y, x = np.ogrid[:shape[0], :shape[1]]  # Create grid of coordinates
        distance = (x - cx)**2 + (y - cy)**2  # Compute squared distances from center
        mask = distance <= radius**2  # Define the circular mask
        channel[mask] = 255  # Apply mask to set values to 255

    # Draw circles around the provided points
    for point in points:
        draw_circle(point, radius)

    return channel


"""
take single-channel image img (numpy array) and add red and green channels highlighting the 2 points
(result has 3-channels)
"""
def add_point_channels(img, point1, point2, radius = 2):
    # Create a channel to visualise the points
    shape = img.shape
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    point1_channel = create_channel([(x1, y1)], shape, radius = radius)
    point2_channel = create_channel([(x2, y2)], shape, radius = radius)

    # Turn one-channel simplified image to 3 channels
    rgb = np.stack([img] * 3, axis=-1)
        
    # Integrate our channel that represents the points into the image
    red_channel = rgb[:, :, 0]
    red_channel = np.maximum(red_channel, point1_channel)
    rgb[:, :, 0] = red_channel

    green_channel = rgb[:, :, 1]
    green_channel = np.maximum(green_channel, point2_channel)
    rgb[:, :, 1] = green_channel

    return rgb



def add_point_channels_multiplepoints(img, points1, points2, radius = 2):
    # Create a channel to visualise the points
    shape = img.shape
    point1_channel = create_channel([(point[0], point[1]) for point in points1], shape, radius = radius)
    point2_channel = create_channel([(point[0], point[1]) for point in points2], shape, radius = radius)

    # Turn one-channel simplified image to 3 channels
    rgb = np.stack([img] * 3, axis=-1)
        
    # Integrate our channel that represents the points into the image
    red_channel = rgb[:, :, 0]
    red_channel = np.maximum(red_channel, point1_channel)
    rgb[:, :, 0] = red_channel

    green_channel = rgb[:, :, 1]
    green_channel = np.maximum(green_channel, point2_channel)
    rgb[:, :, 1] = green_channel

    return rgb


def load_images_from_folder(rel_path_source, max_images=400):
    """
    Load images from a folder using Matplotlib and NumPy.

    Args:
        rel_path_source (str): The relative path to the folder containing the images.
        max_images (int): Maximum number of images to load. Defaults to 400.

    Returns:
        tuple: A tuple containing:
            - images (list): A list of NumPy arrays representing the images.
            - image_names (list): A list of corresponding image file names.
    """
    images = []
    image_names = []
    abs_path = os.path.abspath(rel_path_source)  # Resolve the relative path to an absolute path
    
    if not os.path.exists(abs_path):
        raise ValueError(f"Path '{rel_path_source}' does not exist.")
    
    # Iterate through all files in the directory
    for idx, file_name in enumerate(os.listdir(abs_path)):
        if idx >= max_images:  # Stop if the maximum number of images is reached
            break
        
        file_path = os.path.join(abs_path, file_name)
        
        # Try to load the file as an image
        try:
            img = mpimg.imread(file_path)  # Load the image as a NumPy array
            if img is not None:
                images.append(img)
                image_names.append(file_name)
        except (IOError, ValueError):
            # Skip files that are not images or cannot be opened
            print(f"Skipping non-image file or unreadable file: {file_name}")
    
    return images, image_names

