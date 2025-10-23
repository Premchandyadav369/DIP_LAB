import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or could not be opened.")

    # Apply Otsu's thresholding
    # The function cv2.threshold returns two values: the threshold that was used,
    # and the thresholded image. We use '_' to ignore the first value.
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Otsu's Thresholding")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform image segmentation using Otsu's thresholding.")
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    args = parser.parse_args()
    main(args.image_path)
