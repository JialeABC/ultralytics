import os
import cv2
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Font Configuration
# ----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------
# Utility Functions
# ----------------------------
def find_content_bbox(img, white_threshold=250):
    """
    Automatically detect non-white content region.
    If your images do not have white borders, this function will simply return the full image.
    """
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    mask = gray < white_threshold

    if not np.any(mask):
        H, W = gray.shape
        return 0, H, 0, W

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    pad = 2
    y1 = max(0, y1 - pad)
    y2 = min(gray.shape[0], y2 + pad + 1)
    x1 = max(0, x1 - pad)
    x2 = min(gray.shape[1], x2 + pad + 1)

    return y1, y2, x1, x2


def calculate_ll_statistics(image_path, wavelet="haar", use_bbox=True):
    """
    Calculate LL component statistics for one image.

    Returns:
        dict containing LL statistics.
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32)

    if use_bbox:
        y1, y2, x1, x2 = find_content_bbox(img_np, white_threshold=250)
        img_np = img_np[y1:y2, x1:x2, :]

    # Convert RGB image to grayscale for unified LL statistics
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Make image size even to avoid DWT size mismatch
    h, w = gray.shape
    h = h - h % 2
    w = w - w % 2
    gray = gray[:h, :w]

    # DWT decomposition
    LL, (LH, HL, HH) = pywt.dwt2(gray, wavelet)

    # LL statistics
    ll_min = np.min(LL)
    ll_max = np.max(LL)
    ll_mean = np.mean(LL)
    ll_std = np.std(LL)
    ll_abs_mean = np.mean(np.abs(LL))

    # Recommended low-frequency energy definition
    ll_energy_mean = np.mean(LL ** 2)

    # Sum energy, affected by image size
    ll_energy_sum = np.sum(LL ** 2)

    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "height": h,
        "width": w,
        "LL_shape": str(LL.shape),
        "LL_min": ll_min,
        "LL_max": ll_max,
        "LL_mean": ll_mean,
        "LL_std": ll_std,
        "LL_abs_mean": ll_abs_mean,
        "LL_energy_mean": ll_energy_mean,
        "LL_energy_sum": ll_energy_sum,
    }


def process_folder_ll_statistics(
    image_folder,
    output_csv="ll_statistics.csv",
    output_hist="ll_energy_distribution.png",
    wavelet="haar",
    use_bbox=True
):
    """
    Process all images in a folder and save LL statistics.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        print("No images found in the folder.")
        return

    results = []

    for idx, image_path in enumerate(image_paths):
        try:
            stats = calculate_ll_statistics(
                image_path=image_path,
                wavelet=wavelet,
                use_bbox=use_bbox
            )
            results.append(stats)
            print(f"[{idx + 1}/{len(image_paths)}] Processed: {os.path.basename(image_path)}")

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

    df = pd.DataFrame(results)

    # Save per-image statistics
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved LL statistics to: {output_csv}")

    # Print global summary
    print("\n========== Overall LL Statistics ==========")
    summary_cols = [
        "LL_min",
        "LL_max",
        "LL_mean",
        "LL_std",
        "LL_abs_mean",
        "LL_energy_mean",
        "LL_energy_sum"
    ]

    summary = df[summary_cols].describe()
    print(summary)

    # Save summary as another CSV
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_csv, encoding="utf-8-sig")
    print(f"Saved summary statistics to: {summary_csv}")

    # Plot histogram of LL average energy
    plt.figure(figsize=(8, 6))
    plt.hist(df["LL_energy_mean"], bins=30, edgecolor="black", alpha=0.75)
    plt.xlabel("Mean LL Energy")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Low-Frequency LL Energy")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_hist, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved histogram to: {output_hist}")

    return df


# ----------------------------
# Main Program
# ----------------------------
if __name__ == "__main__":
    image_folder = r"D:\A_my_study\visdrone\yolov5\images"

    output_csv = r"D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/ll_statistics.csv"
    output_hist = r"D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/ll_energy_distribution.png"

    df = process_folder_ll_statistics(
        image_folder=image_folder,
        output_csv=output_csv,
        output_hist=output_hist,
        wavelet="haar",
        use_bbox=True
    )
