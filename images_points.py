import csv
import cv2
from scale import give_length_scale
import os


def load_image(image_path):
    """Load an image in grayscale and color formats."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise cv2.error(f"Failed to load image at path: {image_path}")
    return gray, color


def process_csv_and_add_real_distance(input_csv_path, folder_path, output_image_folder):
    """
    Reads a CSV file, processes images based on the CSV, 
    and saves annotated images in the output image folder.
    
    Args:
        input_csv_path (str): Path to the input CSV file.
        folder_path (str): Path to the folder where input images are stored.
        output_image_folder (str): Path to the folder where annotated images will be saved.
    """
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found at path: {input_csv_path}")

    # Créer le dossier de sortie pour les images annotées
    os.makedirs(output_image_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

    with open(input_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader, None)  # Ignorer les en-têtes

        for row_index, row in enumerate(reader, start=1):
            try:
                filename = row['Image Name']
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                real_distance = row['real_distance']
                point1 = (int(x1), int(y1))
                point2 = (int(x2), int(y2))
                full_image_path = os.path.join(folder_path, filename)

                gray_image, color_image = load_image(full_image_path)
                annotated_image = color_image.copy()

                # Dessiner les points et les annotations
                for idx, point in enumerate([point1, point2], start=1):
                    cv2.circle(annotated_image, point, 5, (0, 255, 0), -1)  # Cercle vert
                    cv2.putText(annotated_image, f"P{idx}", 
                                (point[0] + 10, point[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                3)  # Texte du point (P1 et P2)

                # Récupérer la hauteur et la largeur de l'image
                image_height, image_width, _ = annotated_image.shape

                # Calculer la position du texte "Distance"
                text = f"Distance: {real_distance} mm"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
                text_width, text_height = text_size

                # Afficher le texte en haut à droite de l'image
                cv2.putText(
                    annotated_image, 
                    text, 
                    (image_width - text_width - 20, text_height + 20),  # Position dynamique
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    3
                )

                # Enregistrer l'image annotée dans le dossier de sortie
                annotated_image_path = os.path.join(output_image_folder, filename)
                cv2.imwrite(annotated_image_path, annotated_image)  # Enregistrement de l'image

                print(f"[INFO] Processed row {row_index}: {full_image_path} with points {point1} and {point2}")
                    
            except ValueError as e:
                print(f"[ERROR] Invalid data at row {row_index}: {row} - {e}")
            except FileNotFoundError as e:
                print(f"[ERROR] File not found for row {row_index}: {row} - {e}")
            except cv2.error as e:
                print(f"[ERROR] OpenCV error for row {row_index}: {row} - {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error at row {row_index}: {row} - {e}")


if __name__ == "__main__":
    input_csv_path = "CSVs/massi_annotate.csv"  # Chemin du fichier CSV d'entrée
    folder_path = "/Users/massirashidi/original"  # Dossier contenant les images originales
    output_image_folder = "/Users/massirashidi/original/annotated_images"  # Dossier de sortie des images annotées

    process_csv_and_add_real_distance(input_csv_path, folder_path, output_image_folder)
    print(f"[INFO] Annotated images successfully saved in {output_image_folder}")