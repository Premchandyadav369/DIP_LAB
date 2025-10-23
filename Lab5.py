import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==== USER SETTINGS ====
img_path = r"C:\Users\23BCE7167\Downloads\redbull.jpg"
cutoff = 50
order = 2
# =======================

def spatial_shift(img):
    """Multiply image by (-1)^(x+y) to center spectrum."""
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)[:, None]
    mask = ((x + y) % 2) * -2 + 1
    return img * mask.astype(np.float32)

def create_filter(h, w, cutoff, filter_type="ideal", highpass=False, order=2):
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    D = np.sqrt((x - cx)**2 + (y - cy)**2)

    if filter_type == "ideal":
        H = (D <= cutoff).astype(np.float32)
    elif filter_type == "gaussian":
        H = np.exp(-(D**2) / (2 * cutoff**2))
    elif filter_type == "butterworth":
        H = 1 / (1 + (D / cutoff)**(2 * order))
    else:
        raise ValueError("Unknown filter type")

    if highpass:
        H = 1 - H
    return H.astype(np.float32)

def process_channel(ch, H):
    ch = ch.astype(np.float32)
    shifted = spatial_shift(ch)
    dct_ch = cv2.dct(shifted)
    filtered_dct = dct_ch * H
    inv = cv2.idct(filtered_dct)
    return spatial_shift(inv)

def apply_filter(img, filter_type, highpass, cutoff, order):
    h, w = img.shape[:2]
    H = create_filter(h, w, cutoff, filter_type, highpass, order)
    b, g, r = cv2.split(img)
    b_f = process_channel(b, H)
    g_f = process_channel(g, H)
    r_f = process_channel(r, H)
    result = cv2.merge([b_f, g_f, r_f])

    # Normalize back to [0,255]
    min_val, max_val = result.min(), result.max()
    if max_val > min_val:
        result = (result - min_val) * (255.0 / (max_val - min_val))
    return np.clip(result, 0, 255).astype(np.uint8)

# ---- MAIN ----
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

filter_types = ["ideal", "butterworth", "gaussian"]
highpass_modes = [False, True]  # Lowpass and Highpass

results = [img]
titles = ["Original"]

for ftype in filter_types:
    for hp in highpass_modes:
        filtered_img = apply_filter(img, ftype, hp, cutoff, order)
        results.append(filtered_img)
        titles.append(f"{ftype.capitalize()} {'HPF' if hp else 'LPF'}")

# ---- DISPLAY ----
plt.figure(figsize=(15, 8))
cols = 4
rows = int(np.ceil(len(results) / cols))

for i, (res, title) in enumerate(zip(results, titles)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
