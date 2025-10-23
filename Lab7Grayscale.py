import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

# ===============================
# Load Image (Color)
# ===============================
img = cv2.imread(r"C:\Users\23BCE7167\Downloads\avatarmanga.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256)) / 255.0  # normalize

# ===============================
# Function to Apply Degradation
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

# ===============================
# Process Each Channel Separately
# ===============================
def process_channel(channel):
    H = motion_blur_psf(channel.shape)
    G = fft2(channel) * H
    g = np.abs(ifft2(G))
    noise = np.random.normal(0, 0.01, channel.shape)
    g_noisy = np.clip(g + noise, 0, 1)
    G_noisy = fft2(g_noisy)
    
    # Inverse filter
    inverse_filter = np.real(ifft2(G_noisy / (H + 1e-8)))
    
    # Pseudo-inverse filters
    epsilons = [0.2, 0.02, 0.002, 0.0002]
    pseudo_results = []
    for eps in epsilons:
        H_pseudo = np.where(np.abs(H) > eps, 1 / H, 0)
        pseudo_results.append(np.real(ifft2(G_noisy * H_pseudo)))
    
    # Wiener filter
    K = 0.01
    H_conj = np.conj(H)
    wiener_filter = np.real(ifft2((H_conj / (np.abs(H) ** 2 + K)) * G_noisy))
    
    # CLS filter
    P = np.zeros(channel.shape)
    laplacian = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
    P[:3, :3] = laplacian
    P = np.roll(P, -1, axis=0)
    P = np.roll(P, -1, axis=1)
    P_fft = fft2(P)
    gamma = 0.001
    cls_filter = np.real(ifft2((H_conj / (np.abs(H) ** 2 + gamma * np.abs(P_fft) ** 2)) * G_noisy))
    
    return g_noisy, inverse_filter, pseudo_results, wiener_filter, cls_filter

# Process R, G, B channels separately
channels = cv2.split(img)
results = [process_channel(c) for c in channels]

# Merge channels back
g_noisy = cv2.merge([results[i][0] for i in range(3)])
inverse_filter = cv2.merge([results[i][1] for i in range(3)])
pseudo_results = [cv2.merge([results[i][2][j] for i in range(3)]) for j in range(4)]
wiener_filter = cv2.merge([results[i][3] for i in range(3)])
cls_filter = cv2.merge([results[i][4] for i in range(3)])

titles = ["Original", "Degraded", "Inverse", 
          "Pseudo ε=0.2", "Pseudo ε=0.02", 
          "Pseudo ε=0.002", "Pseudo ε=0.0002", 
          "Wiener", "CLS"]

images = [img, g_noisy, inverse_filter] + pseudo_results + [wiener_filter, cls_filter]

# ===============================
# Display Results in Table Layout
# ===============================
fig, axes = plt.subplots(2, len(images), figsize=(5*len(images), 8))

for i, (ax_img, ax_hist, img_res, title) in enumerate(zip(axes[0], axes[1], images, titles)):
    ax_img.imshow(np.clip(img_res, 0, 1))
    ax_img.set_title(title)
    ax_img.axis("off")
    
    # Plot histogram below each image
    ax_hist.hist(img_res.ravel(), bins=50, color='blue', alpha=0.7)
    ax_hist.set_title(f"{title} Histogram")
    ax_hist.set_yticks([])

plt.tight_layout()
plt.show()
