import cv2
import numpy as np

# Example matrix (instead of image)
A = np.array([[52, 55, 61, 66],
              [70, 61, 64, 73],
              [63, 59, 55, 90],
              [67, 85, 69, 72]], dtype=np.float32)

print("Original Matrix:\n", A)

# ---------------------------
# 1. KL / Hotelling Transform (PCA)
# ---------------------------
X = A.astype(np.float32)

# Perform PCA
mean, eigenvectors = cv2.PCACompute(X, mean=None)

# Project into PCA space
X_pca = cv2.PCAProject(X, mean, eigenvectors)

# Reconstruct from PCA space
X_reconstructed = cv2.PCABackProject(X_pca, mean, eigenvectors)

print("\nKL Transform (Projected Matrix):\n", X_pca)
print("\nKL Transform (Reconstructed Matrix):\n", np.round(X_reconstructed, 2))

# ---------------------------
# 2. Discrete Cosine Transform (DCT)
# ---------------------------
# Normalize for DCT
A_norm = A / 255.0

# Forward DCT
dct = cv2.dct(A_norm)

# Inverse DCT
idct = cv2.idct(dct)

print("\nDCT Coefficients:\n", np.round(dct, 2))
print("\nDCT Reconstructed Matrix:\n", np.round(idct * 255, 2))
