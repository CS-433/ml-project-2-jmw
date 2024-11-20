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
    plt.imshow(image, cmap='gray')
    plt.title("Click to annotate points")
    plt.axis('on')

    # Use ginput to select points manually
    points = plt.ginput(n=2, timeout=0)

    # Convert points to integer coordinates
    points = np.array(points, dtype=int)
    plt.show()

    first_point, second_point = points[0, :], points[1, :]
    
    return first_point, second_point


"""
Write collected data to the CSV file
"""
def write_to_csv(data, output_file):
    # Check if the file already exists
    file_exists = os.path.isfile(output_file)

    # Write header and rows to CSV
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header
        if not file_exists:
            writer.writerow(["Image Name", "Point 1 (x, y)", "Point 2 (x, y)"])
        
        # Write data rows
        for row in data:
            writer.writerow(row)



if __name__ == "__main__":

    rel_path_source = input("Type the relative path of the folder containing the images you want to anotate: \n")
    rel_path_dest = input("Type the relative path of the destination file for the collected data: \n")
    originals, image_names = ip.load_images_from_folder(rel_path_source, max_images=10)

    """
    Could be a problem in the training data if image names aren't unique.
    """
    if len(image_names) != len(get_unique_elements(image_names)):
        raise ValueError(f"Some pictures in {rel_path_source} share the same name !")
    
    collected_data = []
    for i in range(len(originals)):
        # Allow the user to leave every 10 steps. (The last bunch of progress will be lost if he just closes the program.)
        if (i+1) % 10 == 0:
            write_to_csv(collected_data, rel_path_dest)
            collected_data = []
            if input("Do you wish to continue ? Press 'y' to continue, 'n' to stop.") == "n":
                break

        image = originals[i]
        name = image_names[i]
        first_point, second_point = user_annotate(image)

        collected_data.append([name, first_point, second_point])
    
    

