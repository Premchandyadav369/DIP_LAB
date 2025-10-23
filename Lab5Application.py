import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def spatial_shift(img):
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)[:, None]
    mask = ((x + y) % 2) * -2 + 1
    return img * mask.astype(np.float32)

def create_filter(h, w, cutoff, filter_type="ideal", highpass=False, order=2):
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    D = np.sqrt((x - cx)**2 + (y - cy)**2)

    if filter_type == "ideal":
        H = (D <= cutoff).astype(np.float32)
    elif filter_type == "gaussian":
        H = np.exp(-(D**2) / (2 * cutoff**2))
    elif filter_type == "butterworth":
        H = 1 / (1 + (D / cutoff)**(2 * order))
    else:
        raise ValueError("Unknown filter type")

    if highpass:
        H = 1 - H
    return H.astype(np.float32)

def process_grayscale(img, H):
    shifted = spatial_shift(img)
    dct_img = cv2.dct(shifted)
    filtered_dct = dct_img * H
    inv = cv2.idct(filtered_dct)
    return spatial_shift(inv)

def apply_filter_gray(img, filter_type, highpass, cutoff, order):
    h, w = img.shape
    H = create_filter(h, w, cutoff, filter_type, highpass, order)
    result = process_grayscale(img, H)

    # Normalize for better visualization
    min_val, max_val = result.min(), result.max()
    if max_val > min_val:
        result = (result - min_val) * (255.0 / (max_val - min_val))
    return np.clip(result, 0, 255).astype(np.uint8)

def main(image_path, cutoff, order):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Convert to grayscale (luminosity)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Process filters on grayscale
    lpf_result = apply_filter_gray(img_gray, "butterworth", False, cutoff, order)
    hpf_result = apply_filter_gray(img_gray, "butterworth", True, cutoff, order)

    # Display Results
    plt.figure(figsize=(12, 6))
    titles = ["Original (Gray)", "Noise Reduction (Butterworth LPF)", "Detail Enhancement (Butterworth HPF)"]
    images = [img_gray, lpf_result, hpf_result]

    for i, (res, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(res, cmap='gray')
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Butterworth filters to a grayscale image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    parser.add_argument('--cutoff', type=int, default=40, help='Cutoff frequency for the Butterworth filter.')
    parser.add_argument('--order', type=int, default=2, help='Order for the Butterworth filter.')
    args = parser.parse_args()
    main(args.image_path, args.cutoff, args.order)
