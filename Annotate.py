im
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import scipy.signal as signal
import random
import imageProcessing as ip
import csv


"""
This is a simple script that allows a user to provide data by clicking on ant images.
The first click indicates where the "point at which the pronotum meets the cervical shield"
and the second indicates the "posterior basal angle of the metapleuron". The data (image name, 
first point coordinates, second point) is then appended to a CSV file containing the data.
"""


"""
Keep only unique elements from a list
"""


def get_unique_elements(list):
    unique_list = []
    seen = set()

    for item in list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)

    return unique_list


"""
Display image and allow user to annotate coordinates.
"""


def user_annotate(image):
    # Display the image using Matplotlib
    fig = plt.figure(
        figsize=(16, 9)
    )  # Adjust to your screen's aspect ratio or resolution
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Switch to full-screen mode

    plt.imshow(image, cmap="gray")
    plt.title("Click to annotate points")
    plt.axis("on")

    # Use ginput to select points manually
    points = plt.ginput(n=2, timeout=-1)

    # Automatically close the window after getting the points
    plt.close()

    # Convert points to integer coordinates
    points = np.array(points, dtype=int)
    plt.show()

    x1, y1, x2, y2 = points[0, 0], points[0, 1], points[1, 0], points[1, 1]

    return x1, y1, x2, y2


"""
Write collected data to the CSV file
"""


def write_to_csv(data, output_file):
    # Check if the file already exists
    file_exists = os.path.isfile(output_file)

    # Write header and rows to CSV
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write header
        if not file_exists:
            writer.writerow(["Image Name", "x1", "y1", "x2", "y2"])

        # Write data rows
        for row in data:
            # Convert numpy arrays in the row to lists
            processed_row = [
                row[0],  # Keep the string as is
                row[1],
                row[2],
                row[3],
                row[4],
            ]
            writer.writerow(processed_row)


if __name__ == "__main__":

    rel_path_source = input(
        "Type the relative path of the folder containing the images you want to anotate: \n"
    )
    rel_path_dest = input(
        "Type the relative path of the destination file for the collected data: \n"
    )
    originals, image_names = ip.load_images_from_folder(rel_path_source, max_images=400)

    """
    Could be a problem in the training data if image names aren't unique.
    """
    if len(image_names) != len(get_unique_elements(image_names)):
        raise ValueError(f"Some pictures in {rel_path_source} share the same name !")

    collected_data = []
    for i in range(len(originals)):
        # Regularly save the progress
        if (i + 1) % 5 == 0:
            write_to_csv(collected_data, rel_path_dest)
            collected_data = []
            if (i + 1) % 50 == 0:
                if (
                    input(
                        "Do you wish to continue ? Input 'y' to continue, 'n' to stop."
                    )
                    == "n"
                ):
                    break

        image = originals[i]
        name = image_names[i]
        x1, y1, x2, y2 = user_annotate(image)

        collected_data.append([name, x1, y1, x2, y2])
