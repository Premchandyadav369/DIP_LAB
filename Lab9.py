import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst

# ---------------- Step 1: Load and resize color image ----------------
img_path = r"C:\Users\23BCE7167\Downloads\peakyblinders.jpg"  # change path
original = cv2.imread(img_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original, (512, 512))

# ---------------- Step 2: Define compression ratios ----------------
ratios = [5, 10, 25, 50, 75]

def keep_top_left(mat, percent):
    size = int(mat.shape[0] * np.sqrt(percent / 100.0))
    mask = np.zeros_like(mat)
    mask[:size, :size] = 1
    return mat * mask, size

def dst2(a):
    return dst(dst(a.T, type=2).T, type=2)

def idst2(a):
    return idst(idst(a.T, type=2).T, type=2) / (4 * (a.shape[0]) * (a.shape[1]) / (512 * 512))

float_img = np.float32(original)

# ---------------- Step 3: Perform DCT & DST compression ----------------
data_rows = []
dct_images, dst_images = [], []

for r in ratios:
    dct_channels, dst_channels = [], []
    new_size = 0
    for c in range(3):
        # DCT compression
        dct_trans = cv2.dct(float_img[:, :, c])
        dct_masked, size = keep_top_left(dct_trans, r)
        new_size = size
        dct_recon = cv2.idct(dct_masked)
        dct_channels.append(dct_recon)

        # DST compression
        dst_trans = dst2(float_img[:, :, c])
        dst_masked, _ = keep_top_left(dst_trans, r)
        dst_recon = idst2(dst_masked)
        dst_channels.append(dst_recon)

    dct_img = np.clip(cv2.merge(dct_channels), 0, 255).astype(np.uint8)
    dst_img = np.clip(cv2.merge(dst_channels), 0, 255).astype(np.uint8)

    # Save compressed images
    dct_file = f"dct_{r}.jpg"
    dst_file = f"dst_{r}.jpg"
    cv2.imwrite(dct_file, cv2.cvtColor(dct_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(dst_file, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))

    # Get file sizes
    dct_size = os.path.getsize(dct_file) / 1024
    dst_size = os.path.getsize(dst_file) / 1024
    res = f"{new_size}×{new_size}"

    data_rows.append({
        "Compression %": f"{r}%",
        "Resolution": res,
        "DCT Size (KB)": round(dct_size, 2),
        "DST Size (KB)": round(dst_size, 2)
    })

    dct_images.append(dct_img)
    dst_images.append(dst_img)

# ---------------- Step 4: Display results in console ----------------
df = pd.DataFrame(data_rows)
print("\n=== Image Compression Summary (DCT vs DST, 512×512 Input) ===\n")
print(df.to_string(index=False))

# ---------------- Step 5: Display image comparison layout ----------------
rows = len(ratios)
fig, axes = plt.subplots(rows, 2, figsize=(8, 2.5 * rows))
fig.suptitle("DCT vs DST Image Compression (512×512 Input)", fontsize=16)

for i in range(rows):
    # DCT image
    axes[i, 0].imshow(dct_images[i])
    axes[i, 0].set_title(f"DCT {ratios[i]}% ({df['DCT Size (KB)'][i]} KB)")
    axes[i, 0].axis('off')

    # DST image
    axes[i, 1].imshow(dst_images[i])
    axes[i, 1].set_title(f"DST {ratios[i]}% ({df['DST Size (KB)'][i]} KB)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
