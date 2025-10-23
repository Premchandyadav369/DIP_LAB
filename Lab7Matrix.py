import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

def motion_blur_psf(shape, a=0.3, b=0.3):
    M, N = shape
    H = np.zeros((M, N), dtype=np.complex64)
    for u in range(M):
        for v in range(N):
            du = u - M/2
            dv = v - N/2
            val = (du*a + dv*b) * np.pi
            H[u, v] = 1 if val == 0 else (np.sin(val)/val) * np.exp(-1j * val)
    return fftshift(H)

def inverse_filter(F, H): return F / (H + 1e-8)
def pseudo_inverse_filter(F, H, eps):
    H_pseudo = np.where(np.abs(H) > eps, 1/H, 0)
    return F * H_pseudo
def wiener_filter(F, H, K=0.01):
    return (np.conj(H) / (np.abs(H)**2 + K)) * F
def cls_filter(F, H, gamma=0.001):
    P = np.zeros(img.shape)
    P[:3,:3] = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    P = np.roll(P, -1, axis=0)
    P = np.roll(P, -1, axis=1)
    P_fft = fft2(P)
    return (np.conj(H) / (np.abs(H)**2 + gamma*np.abs(P_fft)**2)) * F

def main():
    img = np.array([
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3],
        [0.3, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4],
        [0.4, 0.6, 0.8, 0.9, 1.0, 0.9, 0.7, 0.5],
        [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 0.8, 0.6],
        [0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6],
        [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 0.8, 0.6],
        [0.4, 0.6, 0.8, 0.9, 1.0, 0.9, 0.7, 0.5],
        [0.3, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4],
    ])

    H = motion_blur_psf(img.shape)
    blurred = np.real(ifft2(fft2(img) * H))
    noise = np.random.normal(0, 0.05, img.shape)
    g_noisy = np.clip(blurred + noise, 0, 1)

    F_noisy = fft2(g_noisy)
    inverse_result = np.real(ifft2(inverse_filter(F_noisy, H)))
    pseudo_results = [np.real(ifft2(pseudo_inverse_filter(F_noisy, H, eps)))
                      for eps in [0.2, 0.02, 0.002, 0.0002]]
    wiener_result = np.real(ifft2(wiener_filter(F_noisy, H)))
    cls_result = np.real(ifft2(cls_filter(F_noisy, H)))

    titles = ["Original", "Degraded", "Inverse Filter",
              "Pseudo-Inv (0.2)", "Pseudo-Inv (0.02)",
              "Pseudo-Inv (0.002)", "Pseudo-Inv (0.0002)",
              "Wiener Filter", "CLS Filter"]
    results = [img, g_noisy, inverse_result] + pseudo_results + [wiener_result, cls_result]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    for ax, data, title in zip(axes, results, titles):
        im = ax.imshow(data, cmap="coolwarm")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for (i, j), val in np.ndenumerate(data):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white", weight="bold")
        fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
