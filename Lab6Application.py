import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, color, img_as_float

# === Load Image in COLOR ===
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\og.jpg", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Image not found. Check path!")

# Convert BGR -> RGB and scale to [0,1]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_as_float(img_rgb)

# === Simulate PSF (Point Spread Function) ===
psf = np.ones((5, 5)) / 25.0  # Uniform blur kernel
blurred = cv2.filter2D(img_float, -1, psf)  # Simulate blurred image

# === 1. Wiener Filter ===
def wiener_filter(image, psf, K=0.01):
    image_fft = np.fft.fft2(image, axes=(0, 1))
    psf_fft = np.fft.fft2(psf, s=image.shape[:2])
    psf_fft_conj = np.conj(psf_fft)

    wiener_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + K)
    result_fft = image_fft * wiener_filter[..., np.newaxis]
    result = np.fft.ifft2(result_fft, axes=(0, 1))
    return np.clip(np.abs(result), 0, 1)

wiener_result = wiener_filter(blurred, psf)

# === 2. Tikhonov Regularization Filter ===
def tikhonov_filter(image, psf, alpha=0.01):
    image_fft = np.fft.fft2(image, axes=(0, 1))
    psf_fft = np.fft.fft2(psf, s=image.shape[:2])
    psf_fft_conj = np.conj(psf_fft)

    tikhonov_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + alpha)
    result_fft = image_fft * tikhonov_filter[..., np.newaxis]
    result = np.fft.ifft2(result_fft, axes=(0, 1))
    return np.clip(np.abs(result), 0, 1)

tikhonov_result = tikhonov_filter(blurred, psf)

# === 3. Lucy-Richardson Deconvolution ===
lucy_channels = []
num_iter = 20  # Updated parameter name
for c in range(3):
    lucy_c = restoration.richardson_lucy(blurred[..., c], psf, num_iter=num_iter, clip=False)
    lucy_channels.append(lucy_c)
lucy_result = np.clip(np.stack(lucy_channels, axis=-1), 0, 1)

# === Display Results ===
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
