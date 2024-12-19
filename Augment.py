import albumentations as A
from albumentations.core.keypoints_utils import KeypointParams
from albumentations.augmentations.dropout import coarse_dropout
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from Config import Config


"""
This file creates an augmentation pipeline for images.
"""


"""
Transform the original image without background (R, G, B, Alpha) into single-channel gray-scale.
"""


def quad_channel_2_single_channel(img_path):
    # Read the image with 4 channels (RGBA)
    # Replace the suffix with .png
    base, _ = os.path.splitext(img_path)
    img_path = f"{base}.png"
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has 4 channels
    if image.shape[2] == 4:
        # Split the image into RGB and Alpha channels
        bgr = image[:, :, :3]  # Get the RGB channels (BGR in OpenCV)
        alpha = image[:, :, 3]  # Get the alpha channel

        # Convert the RGB channels to grayscale (using the standard formula for grayscale)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        without_bg = gray * alpha  # Because alpha is 0 for the background

        return without_bg

    else:
        print("The image does not have 4 channels.")


def create_transform_pipeline(augment, model_input_shape):
    if augment:
        return A.Compose(
            [
                # Step 1: Pad the image with zeros to ensure the object stays visible
                A.PadIfNeeded(
                    min_height=1200,  # Minimum height after padding
                    min_width=1600,  # Minimum width after padding
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,  # Padding value (0 for black)
                ),
                # A.Rotate(limit=50, p=1.0, border_mode=cv2.BORDER_CONSTANT),  # Rotate the image
                # A.HorizontalFlip(p=0.5),  # Since most of the images are oriented the same way, we might want to keep it that way so that it is easier for the model
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=30,
                    p=1.0,
                    border_mode=cv2.BORDER_CONSTANT,
                ),  # Combined shift/scale/rotate
                A.CoarseDropout(
                    num_holes_range=(5, 12),
                    hole_height_range=(25, 60),
                    hole_width_range=(25, 60),
                    p=1.0,
                ),  # Randomly mask out patches
                A.GaussNoise(var_limit=(8.0, 40.0), p=0.4),  # Add Gaussian noise
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),  # Adjust brightness/contrast
                A.Blur(blur_limit=3, p=0.3),  # Apply random blur
                # A.ElasticTransform(alpha=1, sigma=50, p=0.2, border_mode=cv2.BORDER_CONSTANT),  # Elastic deformation
                A.Resize(
                    height=model_input_shape[0], width=model_input_shape[1], p=1.0
                ),  # Resize image to fixed shape
            ],
            keypoint_params=A.KeypointParams(format="xy"),  # Keypoints format is (x, y)
        )

    else:
        # When infering, we are just interested in resizing the image to the model input shape.
        return A.Compose(
            [
                A.Resize(
                    height=model_input_shape[0], width=model_input_shape[1], p=1.0
                ),  # Resize image to fixed shape
            ],
            keypoint_params=A.KeypointParams(format="xy"),  # Keypoints format is (x, y)
        )


def reverse_infer_keypoints(keypoints, original_shape, target_shape):
    """
    Reverse the resizing operation on keypoints.

    Args:
        keypoints (list of tuples): List of keypoints in (x, y) format.
        original_shape (tuple): Original image shape (height, width).
        target_shape (tuple): Target image shape (height, width).

    Returns:
        list of tuples: Keypoints transformed back to the original image dimensions.
    """
    orig_height, orig_width = original_shape
    target_height, target_width = target_shape

    scale_factor_x = orig_width / target_width
    scale_factor_y = orig_height / target_height

    # Scale keypoints back
    reversed_keypoints = [
        (kp[0] * scale_factor_x, kp[1] * scale_factor_y) for kp in keypoints
    ]

    return reversed_keypoints


# Get augmented image and keypoints
def prepare_for_model(img_path, pipeline, keypoints=[]):
    single_channel_image = quad_channel_2_single_channel(img_path)
    input_image = pipeline(image=single_channel_image, keypoints=keypoints)
    return (
        input_image["image"],
        input_image["keypoints"],
    )  # if not augment_images else (augmented["image"], [])
