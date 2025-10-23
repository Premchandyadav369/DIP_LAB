import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load grayscale image
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\Sonny Hayes and Ruben.jpg", 0)

# ---------------- Histogram Equalization ----------------
equalized = cv2.equalizeHist(img)

# ---------------- Smoothing Filters ----------------
mean_filtered = cv2.blur(img, (3,3)) # Mean
weighted_filtered = cv2.GaussianBlur(img, (3,3), 0) # Weighted (Gaussian)
median_filtered = cv2.medianBlur(img, 3) # Median

# Mode filter (custom)
def mode_filter(img, ksize=3):
    pad = ksize // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+ksize, j:j+ksize].flatten()
            most_common = Counter(region).most_common(1)[0][0]
            output[i, j] = most_common
    return output

mode_filtered = mode_filter(img, 3)

# Max & Min filters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
max_filtered = cv2.dilate(img, kernel)
min_filtered = cv2.erode(img, kernel)

# ---------------- Sharpening Filters ----------------
# Sobel filter (edge detection)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = np.uint8(sobel)

# Prewitt filter (custom kernels)
kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewitt_x = cv2.filter2D(img, -1, kernelx)
prewitt_y = cv2.filter2D(img, -1, kernely)
prewitt = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))
prewitt = np.uint8(prewitt)

# Laplacian filter
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# ---------------- Display all results ----------------
titles = [
    "Original", "Equalized", "Mean", "Weighted",
    "Median", "Mode", "Max", "Min",
    "Sobel", "Prewitt", "Laplacian"
]
images = [
    img, equalized, mean_filtered, weighted_filtered,
    median_filtered, mode_filtered, max_filtered, min_filtered,
    sobel, prewitt, laplacian
]

plt.figure(figsize=(14,10))
for i in range(len(images)):
    plt.subplot(3,4,i+1) # 3 rows, 4 columns
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()