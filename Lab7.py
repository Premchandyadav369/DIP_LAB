import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import argparse
import os

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

def apply_channelwise_fft_filter(img, H, filter_func, **kwargs):
    restored = np.zeros_like(img)
    for c in range(3):  # R, G, B
        F = fft2(img[..., c])
        restored[..., c] = np.real(ifft2(filter_func(F, H, **kwargs)))
    return np.clip(restored, 0, 1)

def inverse_filter_func(F, H):
    return F / (H + 1e-8)

def pseudo_inverse_func(F, H, eps):
    H_pseudo = np.where(np.abs(H) > eps, 1 / H, 0)
    return F * H_pseudo

def wiener_func(F, H, K):
    H_conj = np.conj(H)
    return (H_conj / (np.abs(H)**2 + K)) * F

def cls_func(F, H, P_fft, gamma):
    H_conj = np.conj(H)
    return (H_conj / (np.abs(H)**2 + gamma * np.abs(P_fft)**2)) * F

def main(image_path, blur_a, blur_b, noise_std, k_wiener, gamma_cls):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0

    H = motion_blur_psf(img.shape[:2], a=blur_a, b=blur_b)

    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[..., c] = np.real(ifft2(fft2(img[..., c]) * H))
    noise = np.random.normal(0, noise_std, img.shape)
    g_noisy = np.clip(blurred + noise, 0, 1)

    inverse_result = apply_channelwise_fft_filter(g_noisy, H, inverse_filter_func)

    epsilons = [0.2, 0.02, 0.002, 0.0002]
    pseudo_results = []
    for eps in epsilons:
        pseudo_results.append(apply_channelwise_fft_filter(g_noisy, H, pseudo_inverse_func, eps=eps))

    wiener_result = apply_channelwise_fft_filter(g_noisy, H, wiener_func, K=k_wiener)

    P = np.zeros(img.shape[:2])
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    P[:3, :3] = laplacian
    P = np.roll(P, -1, axis=0)
    P = np.roll(P, -1, axis=1)
    P_fft = fft2(P)
    cls_result = apply_channelwise_fft_filter(g_noisy, H, cls_func, P_fft=P_fft, gamma=gamma_cls)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Image Restoration Filters.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    parser.add_argument('--blur_a', type=float, default=0.1, help='Motion blur parameter a.')
    parser.add_argument('--blur_b', type=float, default=0.1, help='Motion blur parameter b.')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Standard deviation of Gaussian noise.')
    parser.add_argument('--k_wiener', type=float, default=0.01, help='K value for Wiener filter.')
    parser.add_argument('--gamma_cls', type=float, default=0.001, help='Gamma value for CLS filter.')
    args = parser.parse_args()
    main(args.image_path, args.blur_a, args.blur_b, args.noise_std, args.k_wiener, args.gamma_cls)
