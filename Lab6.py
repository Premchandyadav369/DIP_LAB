# #GrayScale
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # === Load Image ===
# img = cv2.imread(r"C:\Users\23BCE7167\Downloads\og.jpg", cv2.IMREAD_GRAYSCALE)
# if img is None:
#     raise FileNotFoundError("Image not found. Check the path!")

# # === 1. Arithmetic Mean Filter ===
# kernel_size = 3
# arithmetic_mean = cv2.blur(img, (kernel_size, kernel_size))

# # === 2. Geometric Mean Filter ===
# def geometric_mean_filter(img, ksize=3):
#     img_float = img.astype(np.float64) + 1e-6  # avoid log(0)
#     log_img = np.log(img_float)
#     kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)
#     log_mean = cv2.filter2D(log_img, -1, kernel)
#     geo_mean = np.exp(log_mean)
#     return np.uint8(np.clip(geo_mean, 0, 255))

# geometric_mean = geometric_mean_filter(img, kernel_size)

# # === 3. Contraharmonic Mean Filter ===
# def contraharmonic_mean_filter(img, ksize=3, Q=1.5):
#     img_float = img.astype(np.float64)
#     num = cv2.filter2D(np.power(img_float, Q + 1), -1, np.ones((ksize, ksize)))
#     den = cv2.filter2D(np.power(img_float, Q), -1, np.ones((ksize, ksize)))
#     result = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
#     return np.uint8(np.clip(result, 0, 255))

# contraharmonic_q_pos = contraharmonic_mean_filter(img, kernel_size, Q=1.5)
# contraharmonic_q_neg = contraharmonic_mean_filter(img, kernel_size, Q=-1.5)

# # === 4. Alpha-Trimmed Mean Filter ===
# def alpha_trimmed_mean_filter(img, ksize=3, d=2):
#     m, n = ksize, ksize
#     pad = m // 2
#     padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
#     result = np.zeros_like(img, dtype=np.float64)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             window = padded[i:i+m, j:j+n].flatten()
#             window.sort()
#             trimmed = window[d//2:len(window)-d//2]
#             result[i, j] = np.mean(trimmed)
#     return np.uint8(np.clip(result, 0, 255))

# alpha_trimmed_mean = alpha_trimmed_mean_filter(img, ksize=3, d=2)

# # === 5. Min Filter ===
# min_filter = cv2.erode(img, np.ones((kernel_size, kernel_size), np.uint8))

# # === 6. Max Filter ===
# max_filter = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8))

# # === 7. Median Filter ===
# median_filter = cv2.medianBlur(img, kernel_size)

# # === 8. Midpoint Filter ===
# midpoint_filter = ((min_filter.astype(np.float64) + max_filter.astype(np.float64)) / 2).astype(np.uint8)

# # === Plotting All Results Side by Side ===
# titles = ["Original", "Arithmetic Mean", "Geometric Mean",
#           "Contraharmonic Q=+1.5", "Contraharmonic Q=-1.5",
#           "Alpha-Trimmed Mean", "Min Filter", "Max Filter",
#           "Median Filter", "Midpoint Filter"]

# images = [img, arithmetic_mean, geometric_mean,
#           contraharmonic_q_pos, contraharmonic_q_neg,
#           alpha_trimmed_mean, min_filter, max_filter,
#           median_filter, midpoint_filter]

# plt.figure(figsize=(18, 10))
# for i in range(len(images)):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i], fontsize=10)
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

#Colour
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load Image in COLOR ===
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\og.jpg", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")

# Split channels
b, g, r = cv2.split(img)
channels = [b, g, r]

kernel_size = 3

# === 1. Arithmetic Mean Filter ===
arithmetic_channels = [cv2.blur(ch, (kernel_size, kernel_size)) for ch in channels]
arithmetic_mean = cv2.merge(arithmetic_channels)

# === 2. Geometric Mean Filter ===
def geometric_mean_filter(ch, ksize=3):
    ch_float = ch.astype(np.float64) + 1e-6  # avoid log(0)
    log_img = np.log(ch_float)
    kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)
    log_mean = cv2.filter2D(log_img, -1, kernel)
    geo_mean = np.exp(log_mean)
    return np.uint8(np.clip(geo_mean, 0, 255))

geo_channels = [geometric_mean_filter(ch, kernel_size) for ch in channels]
geometric_mean = cv2.merge(geo_channels)

# === 3. Contraharmonic Mean Filter ===
def contraharmonic_mean_filter(ch, ksize=3, Q=1.5):
    ch_float = ch.astype(np.float64)
    num = cv2.filter2D(np.power(ch_float, Q + 1), -1, np.ones((ksize, ksize)))
    den = cv2.filter2D(np.power(ch_float, Q), -1, np.ones((ksize, ksize)))
    result = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return np.uint8(np.clip(result, 0, 255))

contraharmonic_pos_channels = [contraharmonic_mean_filter(ch, kernel_size, Q=1.5) for ch in channels]
contraharmonic_neg_channels = [contraharmonic_mean_filter(ch, kernel_size, Q=-1.5) for ch in channels]
contraharmonic_q_pos = cv2.merge(contraharmonic_pos_channels)
contraharmonic_q_neg = cv2.merge(contraharmonic_neg_channels)

# === 4. Alpha-Trimmed Mean Filter ===
def alpha_trimmed_mean_filter(ch, ksize=3, d=2):
    m, n = ksize, ksize
    pad = m // 2
    padded = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(ch, dtype=np.float64)
    for i in range(ch.shape[0]):
        for j in range(ch.shape[1]):
            window = padded[i:i+m, j:j+n].flatten()
            window.sort()
            trimmed = window[d//2:len(window)-d//2]
            result[i, j] = np.mean(trimmed)
    return np.uint8(np.clip(result, 0, 255))

alpha_channels = [alpha_trimmed_mean_filter(ch, ksize=3, d=2) for ch in channels]
alpha_trimmed_mean = cv2.merge(alpha_channels)

# === 5. Min Filter ===
min_channels = [cv2.erode(ch, np.ones((kernel_size, kernel_size), np.uint8)) for ch in channels]
min_filter = cv2.merge(min_channels)

# === 6. Max Filter ===
max_channels = [cv2.dilate(ch, np.ones((kernel_size, kernel_size), np.uint8)) for ch in channels]
max_filter = cv2.merge(max_channels)

# === 7. Median Filter ===
median_channels = [cv2.medianBlur(ch, kernel_size) for ch in channels]
median_filter = cv2.merge(median_channels)

# === 8. Midpoint Filter ===
midpoint_channels = [((mn.astype(np.float64) + mx.astype(np.float64)) / 2).astype(np.uint8)
                     for mn, mx in zip(min_channels, max_channels)]
midpoint_filter = cv2.merge(midpoint_channels)

# === Plotting All Results Side by Side ===
titles = ["Original", "Arithmetic Mean", "Geometric Mean",
          "Contraharmonic Q=+1.5", "Contraharmonic Q=-1.5",
          "Alpha-Trimmed Mean", "Min Filter", "Max Filter",
          "Median Filter", "Midpoint Filter"]

images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(arithmetic_mean, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(geometric_mean, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(contraharmonic_q_pos, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(contraharmonic_q_neg, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(alpha_trimmed_mean, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(min_filter, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(max_filter, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(midpoint_filter, cv2.COLOR_BGR2RGB)]

plt.figure(figsize=(18, 10))
for i in range(len(images)):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
