"""
Define the software configuration. 
"""

class Config:

    """
    Images will be transformed into this shape (pixel height, pixel width) before being fed to the model.
    We recommend (80, 120). More pixels can allow the model to become more precise but it will require more data.
    """
    input_image_shape_basic_model = (80, 160)
    input_image_shape_pretrained_model = (126, 256)

    """
    The folder that contains ant images without background in .png.
    """
    images_folder_path = "clean"

    """
    The original images (with background and scale)
    """
    original_images_folder_path = "original"

    """
    Training is done on a CSV file that contains 5 columns, for example:
    Image Name,x1,y1,x2,y2
    ...
    casent0904793_p_1.png,333,182,610,172
    ...
    (x1, y1) is the where the Vertex of the Head lies and (x2, y2) is where the Petiole lies (pixel coordinates on the image)
    """
    coords_file_path = "CSVs/combined.csv"
