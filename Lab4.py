import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\TopGun.jpg",cv2.IMREAD_GRAYSCALE)
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
