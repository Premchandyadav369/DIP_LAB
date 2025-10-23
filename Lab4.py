import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found!")

    # ---------------------------
    # 1. KL Transform (Hotelling / PCA)
    # ---------------------------

    # Flatten image into 2D (rows are pixels, columns are features)
    X = img.astype(np.float32)

    # Subtract mean
    mean, eigenvectors = cv2.PCACompute(X, mean=None)

    # Project image data into PCA space
    X_pca = cv2.PCAProject(X, mean, eigenvectors)

    # Reconstruct image back
    X_reconstructed = cv2.PCABackProject(X_pca, mean, eigenvectors)
    kl_img = np.clip(X_reconstructed, 0, 255).astype(np.uint8)

    # ---------------------------
    # 2. Discrete Cosine Transform (DCT)
    # ---------------------------

    # Convert to float32 for DCT
    img_float = np.float32(img) / 255.0

    # Forward DCT
    dct = cv2.dct(img_float)

    # Inverse DCT
    idct = cv2.idct(dct)
    dct_img = np.clip(idct * 255, 0, 255).astype(np.uint8)

    # ---------------------------
    # Display results
    # ---------------------------
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    plt.subplot(1,3,2)
    plt.title("KL Transform (Reconstructed)")
    plt.imshow(kl_img, cmap="gray")

    plt.subplot(1,3,3)
    plt.title("DCT (Reconstructed)")
    plt.imshow(dct_img, cmap="gray")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform KL Transform and DCT on an image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    args = parser.parse_args()
    main(args.image_path)
