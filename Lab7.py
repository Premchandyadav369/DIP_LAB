import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

# ===============================
# Load Image (Color)
# ===============================
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\avatarmanga.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
img = cv2.resize(img, (256, 256))
img = img / 255.0  # normalize

# ===============================
# Degradation Model (Blur + Noise)
# ===============================
def motion_blur_psf(shape, a=0.1, b=0.1, T=1):
    """Generate motion blur PSF in frequency domain"""
    M, N = shape
    H = np.zeros((M, N), dtype=np.complex64)
    for u in range(M):
        for v in range(N):
            du = u - M/2
            dv = v - N/2
            val = (du*a + dv*b) * np.pi
            if val == 0:
                H[u, v] = 1
            else:
                H[u, v] = (np.sin(val) / val) * np.exp(-1j * val)
    return fftshift(H)

H = motion_blur_psf(img.shape[:2])

def apply_channelwise_fft_filter(img, filter_func):
    restored = np.zeros_like(img)
    for c in range(3):  # R, G, B
        F = fft2(img[..., c])
        restored[..., c] = np.real(ifft2(filter_func(F, H)))
    return np.clip(restored, 0, 1)

# Degrade image
blurred = np.zeros_like(img)
for c in range(3):
    blurred[..., c] = np.real(ifft2(fft2(img[..., c]) * H))
noise = np.random.normal(0, 0.01, img.shape)
g_noisy = np.clip(blurred + noise, 0, 1)
G_noisy_channels = [fft2(g_noisy[..., c]) for c in range(3)]

# ===============================
# 1. Inverse Filter
# ===============================
def inverse_filter_func(F, H):
    return F / (H + 1e-8)

inverse_result = apply_channelwise_fft_filter(g_noisy, inverse_filter_func)

# ===============================
# 2. Pseudo-Inverse Filters
# ===============================
epsilons = [0.2, 0.02, 0.002, 0.0002]
pseudo_results = []

for eps in epsilons:
    def pseudo_inverse_func(F, H, eps=eps):
        H_pseudo = np.where(np.abs(H) > eps, 1 / H, 0)
        return F * H_pseudo
    pseudo_results.append(apply_channelwise_fft_filter(g_noisy, pseudo_inverse_func))

# ===============================
# 3. Wiener Filter
# ===============================
K = 0.01
H_conj = np.conj(H)
def wiener_func(F, H):
    return (H_conj / (np.abs(H)**2 + K)) * F

wiener_result = apply_channelwise_fft_filter(g_noisy, wiener_func)

# ===============================
# 4. Constrained Least Squares Filter (Fixed Padding)
# ===============================
# Create Laplacian kernel same size as image
P = np.zeros(img.shape[:2])
laplacian = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
# Place laplacian at top-left corner and roll to center
P[:3, :3] = laplacian
P = np.roll(P, -1, axis=0)
P = np.roll(P, -1, axis=1)

P_fft = fft2(P)
gamma = 0.001

def cls_func(F, H):
    return (H_conj / (np.abs(H)**2 + gamma * np.abs(P_fft)**2)) * F

cls_result = apply_channelwise_fft_filter(g_noisy, cls_func)

# ===============================
# Visualization in Table Layout
# ===============================
titles = [
    "Original", "Degraded", "Inverse Filter",
    "Pseudo-Inv (ε=0.2)", "Pseudo-Inv (ε=0.02)",
    "Pseudo-Inv (ε=0.002)", "Pseudo-Inv (ε=0.0002)",
    "Wiener Filter", "CLS Filter"
]

images = [img, g_noisy, inverse_result] + pseudo_results + [wiener_result, cls_result]

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
axes = axes.ravel()

for ax, image, title in zip(axes, images, titles):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
