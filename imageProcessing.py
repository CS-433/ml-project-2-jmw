from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import scipy.signal as signal
import random



"""
Load a random sample (max_images) from an image folder
"""
def load_images_from_folder(relative_path, max_images=100):
    images = []
    file_names = []
    
    # Get all image file names in the folder
    all_files = [filename for filename in os.listdir(relative_path) if filename.endswith(".jpg") or filename.endswith(".png")]
    
    # Shuffle the list of files
    random.shuffle(all_files)
    
    # Select up to `max_images` files
    selected_files = all_files[:max_images]
    
    for filename in selected_files:
        file_names.append(filename)
        img_path = os.path.join(relative_path, filename)
        img = Image.open(img_path)  # Convert to RGB if needed, e.g., img.convert("RGB")
        img_array = np.array(img)
        images.append(img_array)
    
    return images, file_names


"""
Pad image to the provided target size. The input image should be one-channel.
"""
def pad_image(image, target_size=(1000, 2000)):
    """
    Pad a 2D grayscale image with zeros to the target size.

    Parameters:
        image (numpy array): The original image array with shape (808, x).
        target_size (tuple): The desired size (height, width), e.g., (1000, 2000).

    Returns:
        numpy array: The padded image of size (1000, 2000), 
        transform function: how coordinates x, y are affected.
    """
    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        print("Unpaddable image")
        return np.zeros(target_size), lambda x: x
    
    original_height, original_width = image.shape
    target_height, target_width = target_size

    # Calculate the padding amounts
    pad_height_top = (target_height - original_height) // 2
    pad_height_bottom = target_height - original_height - pad_height_top
    pad_width_left = (target_width - original_width) // 2
    pad_width_right = target_width - original_width - pad_width_left

    # Pad the image with zeros (black pixels)
    padded_image = np.pad(
        image,
        ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
        mode='constant',
        constant_values=0
    )

    # We want to keep track of how coordinates of the original image are modified, because we are using the coords of the crucial points.
    def pad_transform(x, y):
        return x + pad_width_left, y + pad_height_top
    
    return padded_image, pad_transform


"""
Works for the without background images that have 4 channels
(because the last channel indicates the transparancy amound that is
de-activated (0) for the background and activated (1 or 255) for the ant)
"""
def get_black_and_white(image):
    return np.copy(image[:, :, 3])


"""
Works for the without backround images where the 4th channel is tha alpha (transparancy) channel
"""
def to_grayscale(image):
    r, g, b, a = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]

    # Convert to grayscale using the weighted sum method
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # If you want to take the alpha channel into account (e.g., to mask out transparent areas):
    # Here, we are multiplying the grayscale values by the normalized alpha channel (a / 255)
    grayscale = grayscale * (a / 255)

    # Convert the result to an integer type if needed (e.g., uint8 for image data)
    grayscale = grayscale.astype(np.uint8)
    return grayscale


""" Old function
def load_greyscale_images_from_folder(relative_path, max_images = 10):
    images = []
    for filename in os.listdir(relative_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
            img_path = os.path.join(relative_path, filename)
            img = Image.open(img_path).convert("L")  # Convert to RGB if needed
            img_array = np.array(img)
            images.append(img_array)
            if len(images) > max_images:
                return images
    return images
"""


"""
Downsample one channel image by provided factor (factor**2 actually since 2d).
Also return how coords are affected
"""
def downsample(image, factor):
    transform = lambda x, y: (int(x/10), int(y/10))
    return image[:: factor, :: factor], transform


"""
Blur one channel image with avg conv filter of size blur_factor*blur_factor
"""
def blur(image, blur_factor):
    filter = np.ones((blur_factor, blur_factor))/(blur_factor**2)
    res = signal.convolve2d(image, filter, mode='same', boundary='fill', fillvalue=0)
    return res


"""
Remove thin segments (temporary function: should be improved and 130 and 150 should be parameters)
"""
def remove_thin_segments(image, blur_factor):
    blurred = blur(image, blur_factor)
    return np.where((blurred > 30), image, 0).astype(np.uint8)


def simplify_image(original_image):
    #black_and_white = get_black_and_white(original_image)
    grey_scale = to_grayscale(original_image)
    padded, pad_transform = pad_image(grey_scale) # Keep track of fun pad_transform because pad_images affects coordinates.
    # If the image is too big to be padded to size (1000, 2000), pad_image returns an array of 0 by default
    if (padded == 0).all():
        return padded, lambda x: x
    downsampled, ds_transform = downsample(padded, 10)
    #without_thin_segments = remove_thin_segments(downsampled, 15)

    def composed_transform(x, y):
        x2, y2 = pad_transform(x, y)
        return ds_transform(x2, y2)
    return downsampled, composed_transform



def create_channel(point1, point2, shape=(80, 160), radius=2):
    """
    Creates an image channel of the given shape where circles of a given radius
    around point1 and point2 are set to 255, and the rest are zeros.

    Args:
        point1 (tuple): Coordinates (x, y) for the first point.
        point2 (tuple): Coordinates (x, y) for the second point.
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
    draw_circle(point1, radius)
    draw_circle(point2, radius)

    return channel


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



def add_point_channel(img, x1, y1, x2, y2):
    # Create a channel to visualise the points
    point_channel = create_channel((x1, y1), (x2, y2))

    # Turn one-channel simplified image to 3 channels
    rgb = np.stack([img] * 3, axis=-1)
        
    # Integrate our channel that represents the points into the image
    red_channel = rgb[:, :, 0]
    red_channel = np.maximum(red_channel, point_channel)
    rgb[:, :, 0] = red_channel

    return rgb


"""
take single-channel image img (numpy array) and add red and green channels highlighting the 2 points
(result has 3-channels)
"""
def add_point_channels(img, point1, point2):
    # Create a channel to visualise the points
    shape = img.shape
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    point1_channel = create_channel([(x1, y1)], shape, radius = 2)
    point2_channel = create_channel([(x2, y2)], shape, radius = 2)

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


    


