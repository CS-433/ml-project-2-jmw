import csv
from scale import give_length_scale
import os


def process_csv_and_add_real_distance(input_path, output_path, folder_path):
    """
    Reads a CSV file, calculates the real distance for each row using the give_length_scale function,
    and writes a new CSV file with an additional 'real_distance' column.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV file will be saved.
    """
    with open(input_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)

        # Extract header and add new column for 'real_distance'
        headers = next(reader, None)
        if headers:  # If there is a header, add the 'real_distance' column
            headers.append("real_distance")

        # Store updated rows
        updated_rows = [headers] if headers else []

        # Process each row and add the 'real_distance' value
        for row in reader:
            try:
                filename, x1, y1, x2, y2 = row
                point1 = (int(x1), int(y1))
                point2 = (int(x2), int(y2))
                full_image_path = os.path.join(folder_path, filename)

                print(
                    f"Processing row {full_image_path} with points {point1} and {point2}"
                )
                real_distance = give_length_scale(full_image_path, point1, point2)
                if real_distance is not None and real_distance < 100:
                    row.append(str(real_distance))
                    updated_rows.append(row)
                else:
                    row.append("*" + str(real_distance))
                    updated_rows.append(row)

            except ValueError as e:
                print(f"Error processing row {row}: {e}")
            except Exception as e:
                print(f"Unexpected error with row {row}: {e}")

    # Write the updated data into a new CSV file
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)


if __name__ == "__main__":
    input_csv_path = "CSVs/massi.csv"  # Path to your input CSV file
    output_csv_path = "CSVs/massi_annotate.csv"  # Path for the output CSV
    folder_path = (
        "/Users/massirashidi/original"  # Path to the folder containing the images
    )

    process_csv_and_add_real_distance(input_csv_path, output_csv_path, folder_path)
    print(f"File successfully created at {output_csv_path}")
