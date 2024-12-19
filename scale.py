import cv2
import numpy as np
import pytesseract
import re
import csv


def load_image(image_path):
    """Load an image in grayscale and color formats."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return gray, color


def find_horizontal_scale_bar_with_text_color(gray_image, mode="dark"):
    """
    Detect and measure a horizontal scale bar (black or white) with its text.
    """
    if mode == "dark":
        _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    elif mode == "light":
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Invalid mode. Choose 'dark' or 'light'.")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if aspect_ratio > 5:
            horizontal_bars.append((w, x, y, w, h))

    if horizontal_bars:
        horizontal_bars = sorted(horizontal_bars, key=lambda b: b[0], reverse=True)
        selected_bar = horizontal_bars[0]
        x, y, w, h = selected_bar[1:]
        text_region_above = gray_image[max(0, y - 50) : y, x : x + w]
        text_region_below = gray_image[y + h : y + h + 50, x : x + w]
        return selected_bar, text_region_above, text_region_below

    return None, None, None


def validate_text_and_extract_value(scale_text):
    """Validate the text and extract the numeric value if valid."""
    match = re.match(r"^\d+\.?\d*", scale_text)  # Must start with a numeric value
    if match:
        return float(match.group())  # Extract numeric value as float
    return None


def annotate_image_with_scale_and_text(color_image, x, y, w, h, text, point1, point2):
    """Draw a rectangle around the detected scale bar, display the text, and annotate points."""
    annotated_image = color_image.copy()
    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        annotated_image,
        f"Scale: {text}",
        (x, y - 10 if y - 10 > 10 else y + h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    for idx, point in enumerate([point1, point2], start=1):
        cv2.circle(annotated_image, point, 5, (0, 0, 255), -1)
        cv2.putText(
            annotated_image,
            f"P{idx}: {point}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    return annotated_image


def measure_distance_between_points(
    point1, point2, scale_bar_length_pixels, scale_bar_mm
):
    """Calculate the real-world distance between two points given the scale bar information."""
    pixel_distance = np.sqrt(
        (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
    )
    pixel_to_mm_ratio = scale_bar_mm / scale_bar_length_pixels
    return pixel_distance * pixel_to_mm_ratio


def read_scale(text_region_above, text_region_below):
    """Read the text above and below the scale bar."""
    scale_text = pytesseract.image_to_string(
        text_region_above, config="--psm 6"
    ).strip()
    scale_value = validate_text_and_extract_value(scale_text)
    if scale_value is None:
        scale_text = pytesseract.image_to_string(
            text_region_below, config="--psm 6"
        ).strip()
        scale_value = validate_text_and_extract_value(scale_text)
        if scale_value is None:
            raise ValueError(
                f"Could not find a valid scale value. Detected text: {scale_text}"
            )
    return scale_value


def find_horizontal_scale_bar_with_text(gray_image):
    scale_bar_info, text_region_above, text_region_below = (
        find_horizontal_scale_bar_with_text_color(gray_image, mode="dark")
    )
    if not scale_bar_info:
        scale_bar_info, text_region_above, text_region_below = (
            find_horizontal_scale_bar_with_text_color(gray_image, mode="light")
        )

    return scale_bar_info, text_region_above, text_region_below


def give_length_scale(image_path, point1, point2):
    gray_image, color_image = load_image(image_path)
    scale_bar_info, text_region_above, text_region_below = (
        find_horizontal_scale_bar_with_text(gray_image)
    )

    if scale_bar_info:
        scale_bar_length, x, y, w, h = scale_bar_info
        scale_value = read_scale(text_region_above, text_region_below)
        real_distance = measure_distance_between_points(
            point1, point2, scale_bar_length, scale_value
        )
        return real_distance
    else:
        return None


if __name__ == "__main__":
    image_path = "/Users/massirashidi/original/08costa-1723_p_1.jpg"
    point1 = (100, 200)
    point2 = (300, 400)

    real_distance = give_length_scale(image_path, point1, point2)

    print(f"The real-world distance between the two points is: {real_distance} mm.")
