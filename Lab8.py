import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# ---------------- STEP 1: Load Original Image ----------------
original = cv2.imread(r"C:\Users\23BCE7167\Downloads\redbull.jpg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original, (256, 256))

# ---------------- STEP 2: Noise/Degradation Simulation Functions ----------------
def add_gaussian_noise(img, mean=0, var=25):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy

def add_fog(img, intensity=0.5):
    img = img.astype(np.float32)
    fog_layer = 255 * intensity * np.ones_like(img, dtype=np.float32)
    fog = cv2.addWeighted(img, 1 - intensity, fog_layer, intensity, 0)
    return np.clip(fog, 0, 255).astype(np.uint8)

def add_snow(img, intensity=0.3):
    snow = img.copy()
    snow_mask = np.random.rand(*img.shape[:2]) < intensity
    snow[snow_mask] = 255
    return snow

def add_rain(img, drops=500):
    rain = img.copy()
    for _ in range(drops):
        x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
        length = np.random.randint(10, 20)
        cv2.line(rain, (x, y), (x, y + length), (200, 200, 200), 1)
    return rain

def add_demotion(img, scale=0.4):
    h, w = img.shape[:2]
    low_res = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def add_deblur(img, kernel_size=15):
    # motion blur kernel (horizontal)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred

# ---------------- STEP 3: Restoration (Denoising) ----------------
def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# ---------------- STEP 4: Metrics ----------------
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(m))

def ssim_val(img1, img2):
    return ssim(img1, img2, channel_axis=2)

# ---------------- STEP 5: Apply All Degradations ----------------
noises = {
    'Gaussian Noise': add_gaussian_noise(original),
    'Fog': add_fog(original),
    'Snow': add_snow(original),
    'Rain': add_rain(original),
    'Demotion': add_demotion(original),
    'Deblur': add_deblur(original)
}

results = []
images = []

for name, noisy_img in noises.items():
    denoised = denoise_image(noisy_img)

    # Metrics
    mse_noisy = mse(original, noisy_img)
    psnr_noisy = psnr(original, noisy_img)
    ssim_noisy = ssim_val(original, noisy_img)

    mse_denoised = mse(original, denoised)
    psnr_denoised = psnr(original, denoised)
    ssim_denoised = ssim_val(original, denoised)

    results.append({
        'Noise Type': name,
        'MSE (Noisy)': round(mse_noisy, 2),
        'PSNR (Noisy)': round(psnr_noisy, 2),
        'SSIM (Noisy)': round(ssim_noisy, 3),
        'MSE (Denoised)': round(mse_denoised, 2),
        'PSNR (Denoised)': round(psnr_denoised, 2),
        'SSIM (Denoised)': round(ssim_denoised, 3)
    })

    images.append((name, noisy_img, denoised))

# ---------------- STEP 6: Display Summary Table ----------------
df = pd.DataFrame(results)
print("\n📊 Evaluation Summary Table:\n")
print(df.to_string(index=False))

# ---------------- STEP 7: Display Grid Layout ----------------
plt.figure(figsize=(18, 12))
plt.subplot(3, len(images)+1, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

# First row: noisy
for i, (name, noisy, denoised) in enumerate(images):
    plt.subplot(3, len(images)+1, i+2)
    plt.imshow(noisy)
    plt.title(f"{name}\nNoisy")
    plt.axis("off")

# Second row: denoised
for i, (name, noisy, denoised) in enumerate(images):
    plt.subplot(3, len(images)+1, len(images)+i+3)
    plt.imshow(denoised)
    plt.title(f"{name}\nDenoised")
    plt.axis("off")

plt.tight_layout()
plt.show()
