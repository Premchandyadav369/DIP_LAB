import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def geometric_mean_filter(ch, ksize=3):
    ch_float = ch.astype(np.float64) + 1e-6
    log_img = np.log(ch_float)
    kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)
    log_mean = cv2.filter2D(log_img, -1, kernel)
    geo_mean = np.exp(log_mean)
    return np.uint8(np.clip(geo_mean, 0, 255))

def contraharmonic_mean_filter(ch, ksize=3, Q=1.5):
    ch_float = ch.astype(np.float64)
    num = cv2.filter2D(np.power(ch_float, Q + 1), -1, np.ones((ksize, ksize)))
    den = cv2.filter2D(np.power(ch_float, Q), -1, np.ones((ksize, ksize)))
    result = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return np.uint8(np.clip(result, 0, 255))

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

def main(image_path, color_mode, kernel_size, q_pos, q_neg, d_alpha):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    if color_mode:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("Image not found. Check the path!")

        b, g, r = cv2.split(img)
        channels = [b, g, r]

        arithmetic_channels = [cv2.blur(ch, (kernel_size, kernel_size)) for ch in channels]
        arithmetic_mean = cv2.merge(arithmetic_channels)

        geo_channels = [geometric_mean_filter(ch, kernel_size) for ch in channels]
        geometric_mean = cv2.merge(geo_channels)

        contraharmonic_pos_channels = [contraharmonic_mean_filter(ch, kernel_size, Q=q_pos) for ch in channels]
        contraharmonic_neg_channels = [contraharmonic_mean_filter(ch, kernel_size, Q=q_neg) for ch in channels]
        contraharmonic_q_pos = cv2.merge(contraharmonic_pos_channels)
        contraharmonic_q_neg = cv2.merge(contraharmonic_neg_channels)

        alpha_channels = [alpha_trimmed_mean_filter(ch, kernel_size, d=d_alpha) for ch in channels]
        alpha_trimmed_mean = cv2.merge(alpha_channels)

        min_channels = [cv2.erode(ch, np.ones((kernel_size, kernel_size), np.uint8)) for ch in channels]
        min_filter = cv2.merge(min_channels)

        max_channels = [cv2.dilate(ch, np.ones((kernel_size, kernel_size), np.uint8)) for ch in channels]
        max_filter = cv2.merge(max_channels)

        median_channels = [cv2.medianBlur(ch, kernel_size) for ch in channels]
        median_filter = cv2.merge(median_channels)

        midpoint_channels = [((mn.astype(np.float64) + mx.astype(np.float64)) / 2).astype(np.uint8)
                             for mn, mx in zip(min_channels, max_channels)]
        midpoint_filter = cv2.merge(midpoint_channels)

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

    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Image not found. Check the path!")

        arithmetic_mean = cv2.blur(img, (kernel_size, kernel_size))
        geometric_mean = geometric_mean_filter(img, kernel_size)
        contraharmonic_q_pos = contraharmonic_mean_filter(img, kernel_size, Q=q_pos)
        contraharmonic_q_neg = contraharmonic_mean_filter(img, kernel_size, Q=q_neg)
        alpha_trimmed_mean = alpha_trimmed_mean_filter(img, kernel_size, d=d_alpha)
        min_filter = cv2.erode(img, np.ones((kernel_size, kernel_size), np.uint8))
        max_filter = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8))
        median_filter = cv2.medianBlur(img, kernel_size)
        midpoint_filter = ((min_filter.astype(np.float64) + max_filter.astype(np.float64)) / 2).astype(np.uint8)

        images = [img, arithmetic_mean, geometric_mean,
                  contraharmonic_q_pos, contraharmonic_q_neg,
                  alpha_trimmed_mean, min_filter, max_filter,
                  median_filter, midpoint_filter]

    titles = ["Original", "Arithmetic Mean", "Geometric Mean",
              f"Contraharmonic Q=+{q_pos}", f"Contraharmonic Q={q_neg}",
              "Alpha-Trimmed Mean", "Min Filter", "Max Filter",
              "Median Filter", "Midpoint Filter"]

    plt.figure(figsize=(18, 10))
    for i in range(len(images)):
        plt.subplot(2, 5, i+1)
        if color_mode:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply various spatial filters to an image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    parser.add_argument('--color', action='store_true', help='Process the image in color mode.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of the filter kernel.')
    parser.add_argument('--q_pos', type=float, default=1.5, help='Positive Q value for Contraharmonic filter.')
    parser.add_argument('--q_neg', type=float, default=-1.5, help='Negative Q value for Contraharmonic filter.')
    parser.add_argument('--d_alpha', type=int, default=2, help='d value for Alpha-Trimmed Mean filter.')
    args = parser.parse_args()
    main(args.image_path, args.color, args.kernel_size, args.q_pos, args.q_neg, args.d_alpha)
