import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# =========================
# Font setting: Times New Roman
# =========================
rcParams['font.family'] = ['Times New Roman']
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False

# Math font setting
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Font objects
font_title = FontProperties(family='Times New Roman', size=16)
font_label = FontProperties(family='Times New Roman', size=12)
font_legend = FontProperties(family='Times New Roman', size=12)
font_tick = FontProperties(family='Times New Roman', size=11)


def calculate_log_amplitude_histogram(image_path, bins=100):
    """
    Read image, calculate log amplitude spectrum, and return histogram data.
    """
    # 1. Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 2. Perform Fourier transform
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 3. Calculate amplitude spectrum: sqrt(Re^2 + Im^2)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # 4. Calculate log amplitude spectrum: log(1 + magnitude)
    log_magnitude = np.log1p(magnitude)

    # 5. Calculate histogram
    hist, bin_edges = np.histogram(
        log_magnitude,
        bins=bins,
        range=(0, 15),
        density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist


# Image paths
visible_path = 'D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/00213.jpg'
fake_ir_path = 'D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/00213_fake.png'
real_ir_path = 'D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/00213_r.jpg'

# Calculate histogram data
bins = 100
x_vis, y_vis = calculate_log_amplitude_histogram(visible_path, bins)
x_fake, y_fake = calculate_log_amplitude_histogram(fake_ir_path, bins)
x_real, y_real = calculate_log_amplitude_histogram(real_ir_path, bins)

# Plot comparison figure
plt.figure(figsize=(10, 6))

plt.plot(x_vis, y_vis, label='Real Visible Image', color='blue', linewidth=2)
plt.plot(x_fake, y_fake, label='Synthetic Infrared Image (HSLE)', color='green', linewidth=2, linestyle='--')
plt.plot(x_real, y_real, label='Real Infrared Image', color='red', linewidth=2, linestyle=':')

plt.title('Comparison of Log Amplitude Spectrum Histograms', fontproperties=font_title)
plt.xlabel('Log Amplitude Value', fontproperties=font_label)
plt.ylabel('Probability Density', fontproperties=font_label)

plt.legend(prop=font_legend)

# Set tick font
plt.xticks(fontproperties=font_tick)
plt.yticks(fontproperties=font_tick)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
