#pythonprogram
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the image in grayscale
image_path = r"C:\Users\23BCE7167\Downloads\Sonny Hayes and Ruben.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError("Image path not found. Please check the path.")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(5, 5))
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
# 1. Image Negative
negative_img = 255 - img
plt.title("Negative Image")
plt.imshow(negative_img, cmap='gray')
plt.axis('off')
plt.show()
# 2. Log Transformation
img_float = img.astype(np.float32)
c = 255 / np.log(1 + np.max(img_float))
log_image = c * np.log(1 + img_float)
log_image = np.uint8(np.clip(log_image, 0, 255))
plt.title("Log Transformation")
plt.imshow(log_image, cmap='gray')
plt.axis('off')
plt.show()
# 3. Gamma (Power-law) Transformation
gamma = 0.5  # You can try 2.0 as well
c = 255.0 / (np.max(img) ** gamma)
gamma_img = c * (img.astype(np.float32) ** gamma)
gamma_img = np.uint8(np.clip(gamma_img, 0, 255))
plt.title(f"Gamma Transformation (γ = {gamma})")
plt.imshow(gamma_img, cmap='gray')
plt.axis('off')
plt.show()
# 4. Piecewise Linear Transformation
def piecewise_linear(r):
    r1, s1 = 70, 0
    r2, s2 = 140, 255
    return np.piecewise(
        r,
        [r < r1, (r >= r1) & (r <= r2), r > r2],
        [lambda r: s1 / r1 * r,
         lambda r: ((s2 - s1) / (r2 - r1)) * (r - r1) + s1,
         lambda r: ((255 - s2) / (255 - r2)) * (r - r2) + s2]
    )

piecewise_img = piecewise_linear(img).astype(np.uint8)
plt.title("Piecewise Linear Transformation")
plt.imshow(piecewise_img, cmap='gray')
plt.axis('off')
plt.show()

# 5. Bit-plane Slicing
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle('Bit-plane Slicing', fontsize=16)

for i in range(8):
    bit_img = np.bitwise_and(img, 1 << i)
    bit_img = np.where(bit_img > 0, 255, 0).astype(np.uint8)
    ax = axes[i//4, i%4]
    ax.imshow(bit_img, cmap='gray')
    ax.set_title(f'Bit Plane {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()
