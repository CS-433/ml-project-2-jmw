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
Pad image to the provided target size. The input image should be one-channel
"""
def pad_image(image, target_size=(1000, 2000)):
    """
    Pad a 2D grayscale image with zeros to the target size.

    Parameters:
        image (numpy array): The original image array with shape (808, x).
        target_size (tuple): The desired size (height, width), e.g., (1000, 2000).

    Returns:
        numpy array: The padded image of size (1000, 2000).
    """
    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        print("Unpaddable image")
        return None
    
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
    
    return padded_image


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
Downsample one channel image by provided factor (factor**2 actually since 2d)
"""
def downsample(image, factor):
    return image[:: factor, :: factor]


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
    return np.where((image > 130) & (blurred > 150), 1, 0).astype(np.uint8)


