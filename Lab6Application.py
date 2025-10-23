import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, color, img_as_float
import argparse
import os

def wiener_filter(image, psf, K=0.01):
    image_fft = np.fft.fft2(image, axes=(0, 1))
    psf_fft = np.fft.fft2(psf, s=image.shape[:2])
    psf_fft_conj = np.conj(psf_fft)

    wiener_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + K)
    result_fft = image_fft * wiener_filter[..., np.newaxis]
    result = np.fft.ifft2(result_fft, axes=(0, 1))
    return np.clip(np.abs(result), 0, 1)

def tikhonov_filter(image, psf, alpha=0.01):
    image_fft = np.fft.fft2(image, axes=(0, 1))
    psf_fft = np.fft.fft2(psf, s=image.shape[:2])
    psf_fft_conj = np.conj(psf_fft)

    tikhonov_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + alpha)
    result_fft = image_fft * tikhonov_filter[..., np.newaxis]
    result = np.fft.ifft2(result_fft, axes=(0, 1))
    return np.clip(np.abs(result), 0, 1)

def main(image_path, k_wiener, alpha_tikhonov, iter_lucy):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found. Check path!")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    psf = np.ones((5, 5)) / 25.0
    blurred = cv2.filter2D(img_float, -1, psf)

    wiener_result = wiener_filter(blurred, psf, K=k_wiener)
    tikhonov_result = tikhonov_filter(blurred, psf, alpha=alpha_tikhonov)

    lucy_channels = []
    for c in range(3):
        lucy_c = restoration.richardson_lucy(blurred[..., c], psf, num_iter=iter_lucy, clip=False)
        lucy_channels.append(lucy_c)
    lucy_result = np.clip(np.stack(lucy_channels, axis=-1), 0, 1)

    titles = ["Original Image", "Wiener Filter", "Tikhonov Regularization", "Lucy-Richardson"]
    images = [img_float, wiener_result, tikhonov_result, lucy_result]

    plt.figure(figsize=(14, 6))
    for i in range(len(images)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply advanced image restoration techniques.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    parser.add_argument('--k_wiener', type=float, default=0.01, help='K value for Wiener filter.')
    parser.add_argument('--alpha_tikhonov', type=float, default=0.01, help='Alpha value for Tikhonov regularization.')
    parser.add_argument('--iter_lucy', type=int, default=20, help='Number of iterations for Lucy-Richardson deconvolution.')
    args = parser.parse_args()
    main(args.image_path, args.k_wiener, args.alpha_tikhonov, args.iter_lucy)
