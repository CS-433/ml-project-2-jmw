import albumentations as A
from albumentations.core.keypoints_utils import KeypointParams
from albumentations.augmentations.dropout import coarse_dropout
import numpy as np
import cv2
from matplotlib import pyplot as plt



def quad_channel_2_single_channel(img_path):
    # Read the image with 4 channels (RGBA)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has 4 channels
    if image.shape[2] == 4:
        # Split the image into RGB and Alpha channels
        bgr = image[:, :, :3]  # Get the RGB channels (BGR in OpenCV)
        alpha = image[:, :, 3]  # Get the alpha channel

        # Convert the RGB channels to grayscale (using the standard formula for grayscale)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        without_bg = gray*alpha

        return without_bg

    else:
        print("The image does not have 4 channels.")


# Define an augmentation pipeline
transform = A.Compose(
    [
        A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),  # Rotate the image
        # A.HorizontalFlip(p=0.5),  # Flip the image
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0, border_mode=cv2.BORDER_CONSTANT),  # Combined shift/scale/rotate
        A.CoarseDropout(num_holes_range=(3, 10), hole_height_range=(50, 100), hole_width_range=(40, 80), fill="random_uniform", p=1.0),  # Randomly mask out patches
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),  # Add Gaussian noise
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness/contrast
        A.Blur(blur_limit=3, p=0.3),  # Apply random blur
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2, border_mode=cv2.BORDER_CONSTANT),  # Elastic deformation
        A.Resize(height=50, width=100, p=1.0),  # Resize image to fixed shape (w, h) - Change (w, h) as per your needs
        #A.Lambda(image=lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), p=1.0)  # Convert RGB to grayscale
    ],

    keypoint_params=A.KeypointParams(format="xy")  # Keypoints format is (x, y)
)


def prepare_for_model(img_path, keypoints):
    single_channel_image = quad_channel_2_single_channel(img_path)
    augmented = transform(image = single_channel_image, keypoints = keypoints)
    return augmented["image"], augmented["keypoints"]


if __name__ == "__main__":
    augmented, keypoints = prepare_for_model("clean/anic32-032237-2_p_1.png", [(100, 100), (200, 100)])

    plt.plot()
    plt.imshow(augmented, cmap = "gray")
    plt.show()

    print("Transformed Keypoints:", keypoints)
    print(augmented.shape)

