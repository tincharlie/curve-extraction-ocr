import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import os
from matplotlib import pyplot as plt

# Config
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this if needed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# === Step 1: Convert PDF to image(s) ===
def pdf_to_images(pdf_path, output_dir="output_images"):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths


# === Step 2: Preprocess Image ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return image, edges


# === Step 3: Extract Curve Points ===
def extract_curve_points(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for cnt in contours:
        for p in cnt:
            x, y = p[0]
            points.append((x, y))
    return points


# === Step 4: Axis Calibration ===
def get_axis_scaling(image, pixel_ref_points, value_ref_points):
    (x1_px, y1_px), (x2_px, y2_px) = pixel_ref_points
    (x1_val, y1_val), (x2_val, y2_val) = value_ref_points

    scale_x = (x2_val - x1_val) / (x2_px - x1_px)
    scale_y = (y2_val - y1_val) / (y2_px - y1_px)
    offset_x = x1_val - x1_px * scale_x
    offset_y = y1_val - y1_px * scale_y
    return scale_x, offset_x, scale_y, offset_y


# === Step 5: Scale pixel points ===
def scale_points(pixel_points, scale_x, offset_x, scale_y, offset_y):
    return [(x * scale_x + offset_x, y * scale_y + offset_y) for x, y in pixel_points]


# === Step 6: Save to CSV ===
def save_to_csv(scaled_points, filename="curve_points.csv"):
    df = pd.DataFrame(scaled_points, columns=["X", "Y"])
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} points to {filename}")


# === Main Function to Handle PDF or Image ===
def process_file(file_path, is_pdf=True):
    if is_pdf:
        image_paths = pdf_to_images(file_path)
    else:
        image_paths = [file_path]  # Directly use the image

    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        image, edges = preprocess_image(img_path)
        points = extract_curve_points(edges)

        # === MANUAL: Update these with real pixel/value pairs from axes ===
        pixel_refs = [(100, 400), (400, 100)]   # pixel locations on image
        value_refs = [(0, 0), (100, 100)]       # real-world values

        scale_x, offset_x, scale_y, offset_y = get_axis_scaling(image, pixel_refs, value_refs)
        scaled = scale_points(points, scale_x, offset_x, scale_y, offset_y)

        csv_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_curve.csv"
        save_to_csv(scaled, filename=csv_name)

        # Optional Visualization
        plt.imshow(image)
        xs, ys = zip(*points)
        plt.scatter(xs, ys, s=1, c='r')
        plt.title(f"Extracted Curve: {os.path.basename(img_path)}")
        plt.show()


# === Example Usage ===
if __name__ == "__main__":
    # Provide your file path below
    input_path = "sample_curve.pdf"   # or "curve_image.jpg"
    is_pdf = input_path.lower().endswith(".pdf")

    process_file(input_path, is_pdf=is_pdf)
