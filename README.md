# DIP Lab Experiments

This repository contains a series of Python scripts for Digital Image Processing (DIP) experiments. Each script explores fundamental concepts and techniques in image processing.

**Done by: V C Premchand Yadav**

---

## Lab 2: Image Enhancement in the Spatial Domain

This script focuses on point processing techniques for image enhancement.

### Techniques Used:
- **Image Negative:** Inverts the intensity levels of an image.
- **Log Transformation:** Compresses the dynamic range of an image, enhancing details in darker regions.
- **Gamma (Power-law) Transformation:** Adjusts the brightness and contrast of an image.
- **Piecewise Linear Transformation:** Applies a linear transformation in a piecewise manner to enhance specific intensity ranges.
- **Bit-plane Slicing:** Separates the image into its 8 bit-planes to show the contribution of each bit to the final image.

---

## Lab 3: Spatial Filtering and Histogram Equalization

This script demonstrates histogram equalization and various spatial filtering techniques for smoothing and sharpening.

### Techniques Used:
- **Histogram Equalization:** Improves contrast by redistributing pixel intensities.
- **Smoothing Filters:**
  - **Mean Filter:** Averages pixel values in a neighborhood.
  - **Weighted (Gaussian) Filter:** Applies a weighted average, giving more importance to central pixels.
  - **Median Filter:** Replaces each pixel with the median of its neighbors, effective for salt-and-pepper noise.
  - **Mode, Max & Min Filters:** Use the mode, maximum, or minimum value in the neighborhood.
- **Sharpening Filters:**
  - **Sobel & Prewitt Filters:** First-order derivative filters for edge detection.
  - **Laplacian Filter:** A second-order derivative filter for edge detection.

---

## Lab 4: Image Transforms

This lab explores two fundamental image transforms: the Karhunen-Loève (KL) Transform and the Discrete Cosine Transform (DCT).

### `Lab4.1.py`
Demonstrates the KL Transform (also known as Hotelling Transform or PCA) and DCT on a sample matrix.

### `Lab4.py`
Applies the KL Transform and DCT to a grayscale image and reconstructs it to show the effect of these transforms.

---

## Lab 5: Image Filtering in the Frequency Domain

This lab focuses on filtering techniques in the frequency domain.

### `Lab5.py`
Implements Ideal, Butterworth, and Gaussian filters for both lowpass and highpass filtering on color images.

### `Lab5Application.py`
Applies Butterworth lowpass and highpass filters to a grayscale image to demonstrate noise reduction and detail enhancement.

---

## Lab 6: Image Restoration and Denoising

This lab explores various techniques for image restoration from degradation.

### `Lab6.py`
Implements a wide range of spatial filters for noise reduction on both grayscale and color images, including:
- Arithmetic and Geometric Mean Filters
- Contraharmonic and Alpha-Trimmed Mean Filters
- Min, Max, Median, and Midpoint Filters

### `Lab6Application.py`
Demonstrates advanced image restoration techniques like the Wiener filter, Tikhonov regularization, and Lucy-Richardson deconvolution.

---

## Lab 7: Advanced Image Restoration Filters

This lab implements and visualizes several advanced filters for restoring degraded images.

### `Lab7.py`
Applies Inverse, Pseudo-Inverse, Wiener, and Constrained Least Squares (CLS) filters to a color image with simulated motion blur and noise.

### `Lab7Grayscale.py`
Processes each color channel of an image separately with the same set of restoration filters and visualizes the results with histograms.

### `Lab7Matrix.py`
Demonstrates the effect of these restoration filters on a sample matrix, providing a numerical view of their behavior.

---

## Lab 8: Image Degradation, Restoration, and Quality Metrics

This script simulates various image degradations, applies a denoising algorithm, and evaluates the results using objective quality metrics.

### Degradations Simulated:
- Gaussian Noise, Fog, Snow, Rain
- Demotion (downscaling and upscaling)
- Deblur (motion blur)

### Metrics Used:
- **MSE (Mean Squared Error):** Measures the average squared difference between the estimated and original images.
- **PSNR (Peak Signal-to-Noise Ratio):** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
- **SSIM (Structural Similarity Index):** Measures the similarity between two images.

---

## Lab 9: Image Compression

This script compares the performance of DCT (Discrete Cosine Transform) and DST (Discrete Sine Transform) for image compression at various compression ratios.

### Process:
1. The input image is transformed using both DCT and DST.
2. A certain percentage of the transform coefficients (from the top-left) are kept, and the rest are discarded.
3. The image is reconstructed from the remaining coefficients.
4. The compressed image sizes are compared and displayed in a summary table.
