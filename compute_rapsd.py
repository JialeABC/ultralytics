# # import os
# # from pathlib import Path
# #
# # import cv2
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# #
# # def read_image_gray(path):
# #     """
# #     Read image as grayscale float32.
# #     """
# #     img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
# #
# #     if img is None:
# #         raise FileNotFoundError(f"Cannot read image: {path}")
# #
# #     # If RGB / BGR image, convert to grayscale
# #     if img.ndim == 3:
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# #     img = img.astype(np.float32)
# #
# #     # Normalize different bit-depth images to [0, 1]
# #     if img.max() > 1.0:
# #         img = img / img.max()
# #
# #     return img
# #
# #
# # def compute_rapsd(img, num_bins=128, eps=1e-8, use_hann_window=True):
# #     """
# #     Compute Radial Average Power Spectral Density (RAPSD).
# #
# #     Args:
# #         img: 2D grayscale image.
# #         num_bins: number of radial frequency bins.
# #         eps: numerical stability term.
# #         use_hann_window: whether to apply Hann window before FFT.
# #
# #     Returns:
# #         rapsd_norm: normalized RAPSD vector, shape [num_bins].
# #     """
# #
# #     img = img.astype(np.float32)
# #
# #     # Remove DC component
# #     img = img - np.mean(img)
# #
# #     h, w = img.shape
# #
# #     # Apply Hann window to reduce boundary artifacts
# #     if use_hann_window:
# #         win_y = np.hanning(h)
# #         win_x = np.hanning(w)
# #         window = np.outer(win_y, win_x)
# #         img = img * window
# #
# #     # 2D FFT
# #     fft = np.fft.fft2(img)
# #     fft_shift = np.fft.fftshift(fft)
# #
# #     # Power spectrum
# #     power = np.abs(fft_shift) ** 2
# #
# #     # Radial distance map
# #     cy, cx = h // 2, w // 2
# #     y, x = np.indices((h, w))
# #     r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
# #
# #     # Normalize radius to [0, 1]
# #     r = r / (r.max() + eps)
# #
# #     # Radial bins
# #     bins = np.linspace(0, 1, num_bins + 1)
# #     rapsd = np.zeros(num_bins, dtype=np.float32)
# #
# #     for i in range(num_bins):
# #         mask = (r >= bins[i]) & (r < bins[i + 1])
# #         if np.any(mask):
# #             rapsd[i] = power[mask].mean()
# #
# #     # Normalize RAPSD as a distribution
# #     rapsd_norm = rapsd / (np.sum(rapsd) + eps)
# #
# #     return rapsd_norm
# #
# #
# # def log_rapsd_distance(rapsd_a, rapsd_b, eps=1e-8):
# #     """
# #     Compute log-RAPSD distance between two RAPSD curves.
# #     """
# #     return np.mean(np.abs(np.log(rapsd_a + eps) - np.log(rapsd_b + eps)))
# #
# #
# # def get_image_files(folder):
# #     """
# #     Get image files in a folder.
# #     """
# #     exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
# #     folder = Path(folder)
# #
# #     files = []
# #     for ext in exts:
# #         files.extend(folder.glob(f"*{ext}"))
# #         files.extend(folder.glob(f"*{ext.upper()}"))
# #
# #     return sorted(files)
# #
# #
# # def match_pairs_by_stem(ir_folder, vis_folder):
# #     """
# #     Match infrared and visible images by filename stem.
# #     Example:
# #         infrared/0001.png
# #         visible/0001.jpg
# #     will be matched because both stems are '0001'.
# #     """
# #     ir_files = get_image_files(ir_folder)
# #     vis_files = get_image_files(vis_folder)
# #
# #     ir_dict = {p.stem: p for p in ir_files}
# #     vis_dict = {p.stem: p for p in vis_files}
# #
# #     common_stems = sorted(set(ir_dict.keys()) & set(vis_dict.keys()))
# #
# #     pairs = []
# #     for stem in common_stems:
# #         pairs.append((ir_dict[stem], vis_dict[stem]))
# #
# #     missing_ir = sorted(set(vis_dict.keys()) - set(ir_dict.keys()))
# #     missing_vis = sorted(set(ir_dict.keys()) - set(vis_dict.keys()))
# #
# #     print(f"Found IR images: {len(ir_files)}")
# #     print(f"Found VIS images: {len(vis_files)}")
# #     print(f"Matched pairs: {len(pairs)}")
# #
# #     if len(missing_ir) > 0:
# #         print(f"Warning: {len(missing_ir)} visible images have no matched infrared image.")
# #
# #     if len(missing_vis) > 0:
# #         print(f"Warning: {len(missing_vis)} infrared images have no matched visible image.")
# #
# #     return pairs
# #
# #
# # def compute_dataset_rapsd(
# #     ir_folder,
# #     vis_folder,
# #     output_dir,
# #     num_bins=128,
# #     resize_to_ir=True,
# #     eps=1e-8
# # ):
# #     output_dir = Path(output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)
# #
# #     pairs = match_pairs_by_stem(ir_folder, vis_folder)
# #
# #     if len(pairs) == 0:
# #         raise RuntimeError("No matched image pairs found. Please check file names.")
# #
# #     ir_rapsd_list = []
# #     vis_rapsd_list = []
# #     distance_list = []
# #
# #     per_image_records = []
# #
# #     for idx, (ir_path, vis_path) in enumerate(pairs):
# #         ir = read_image_gray(ir_path)
# #         vis = read_image_gray(vis_path)
# #
# #         # Make sure paired images have the same size
# #         if resize_to_ir:
# #             h, w = ir.shape
# #             vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)
# #         else:
# #             h, w = vis.shape
# #             ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_AREA)
# #
# #         ir_rapsd = compute_rapsd(ir, num_bins=num_bins, eps=eps)
# #         vis_rapsd = compute_rapsd(vis, num_bins=num_bins, eps=eps)
# #
# #         dist = log_rapsd_distance(vis_rapsd, ir_rapsd, eps=eps)
# #
# #         ir_rapsd_list.append(ir_rapsd)
# #         vis_rapsd_list.append(vis_rapsd)
# #         distance_list.append(dist)
# #
# #         record = {
# #             "index": idx,
# #             "infrared_name": ir_path.name,
# #             "visible_name": vis_path.name,
# #             "log_RAPSD_distance_VIS_IR": dist,
# #         }
# #
# #         for b in range(num_bins):
# #             record[f"IR_bin_{b}"] = ir_rapsd[b]
# #             record[f"VIS_bin_{b}"] = vis_rapsd[b]
# #
# #         per_image_records.append(record)
# #
# #         print(
# #             f"[{idx + 1}/{len(pairs)}] "
# #             f"{ir_path.name} | {vis_path.name} | "
# #             f"log-RAPSD distance = {dist:.6f}"
# #         )
# #
# #     ir_rapsd_array = np.stack(ir_rapsd_list, axis=0)
# #     vis_rapsd_array = np.stack(vis_rapsd_list, axis=0)
# #     distance_array = np.array(distance_list)
# #
# #     # Mean and std RAPSD
# #     ir_mean = ir_rapsd_array.mean(axis=0)
# #     ir_std = ir_rapsd_array.std(axis=0)
# #
# #     vis_mean = vis_rapsd_array.mean(axis=0)
# #     vis_std = vis_rapsd_array.std(axis=0)
# #
# #     freq = np.linspace(0, 1, num_bins)
# #
# #     # Save mean RAPSD
# #     mean_df = pd.DataFrame({
# #         "normalized_frequency": freq,
# #         "IR_mean_RAPSD": ir_mean,
# #         "IR_std_RAPSD": ir_std,
# #         "VIS_mean_RAPSD": vis_mean,
# #         "VIS_std_RAPSD": vis_std,
# #     })
# #
# #     mean_csv_path = output_dir / "mean_rapsd.csv"
# #     mean_df.to_csv(mean_csv_path, index=False)
# #
# #     # Save per-image RAPSD
# #     per_image_df = pd.DataFrame(per_image_records)
# #     per_image_csv_path = output_dir / "per_image_rapsd.csv"
# #     per_image_df.to_csv(per_image_csv_path, index=False)
# #
# #     # Save summary
# #     summary_df = pd.DataFrame({
# #         "metric": [
# #             "number_of_pairs",
# #             "mean_log_RAPSD_distance_VIS_IR",
# #             "std_log_RAPSD_distance_VIS_IR",
# #         ],
# #         "value": [
# #             len(pairs),
# #             distance_array.mean(),
# #             distance_array.std(),
# #         ]
# #     })
# #
# #     summary_csv_path = output_dir / "summary.csv"
# #     summary_df.to_csv(summary_csv_path, index=False)
# #
# #     # Plot mean RAPSD curve
# #     plt.figure(figsize=(7, 5))
# #
# #     plt.plot(freq, np.log(ir_mean + eps), label="Infrared")
# #     plt.fill_between(
# #         freq,
# #         np.log(ir_mean + ir_std + eps),
# #         np.log(np.maximum(ir_mean - ir_std, eps)),
# #         alpha=0.2
# #     )
# #
# #     plt.plot(freq, np.log(vis_mean + eps), label="Visible")
# #     plt.fill_between(
# #         freq,
# #         np.log(vis_mean + vis_std + eps),
# #         np.log(np.maximum(vis_mean - vis_std, eps)),
# #         alpha=0.2
# #     )
# #
# #     plt.xlabel("Normalized radial frequency")
# #     plt.ylabel("Log normalized RAPSD")
# #     plt.title("Mean RAPSD Curves")
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #
# #     fig_path = output_dir / "rapsd_mean_curve.png"
# #     plt.savefig(fig_path, dpi=300)
# #     plt.close()
# #
# #     print("\nDone.")
# #     print(f"Mean RAPSD saved to: {mean_csv_path}")
# #     print(f"Per-image RAPSD saved to: {per_image_csv_path}")
# #     print(f"Summary saved to: {summary_csv_path}")
# #     print(f"Figure saved to: {fig_path}")
# #
# #     print("\nSummary:")
# #     print(f"Number of pairs: {len(pairs)}")
# #     print(f"Mean log-RAPSD distance VIS-IR: {distance_array.mean():.6f}")
# #     print(f"Std log-RAPSD distance VIS-IR: {distance_array.std():.6f}")
# #
# #
# # if __name__ == "__main__":
# #     infrared_folder = r"D:\A_my_study\visdrone\rapsd_inf"
# #     visible_folder = r"D:\A_my_study\visdrone\rapsd_vis"
# #     output_folder = r"D:\A_my_study\visdrone/rapsd_results"
# #
# #     compute_dataset_rapsd(
# #         ir_folder=infrared_folder,
# #         vis_folder=visible_folder,
# #         output_dir=output_folder,
# #         num_bins=128,
# #         resize_to_ir=True
# #     )
#
#
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def plot_rapsd_from_csv(
#     csv_path,
#     output_dir,
#     output_name="rapsd_curve_from_csv.png",
#     eps=1e-8,
#     show_std=True,
#     log_y=True
# ):
#     """
#     Plot mean RAPSD curves from a CSV file.
#
#     Required columns:
#         normalized_frequency
#         IR_mean_RAPSD
#         IR_std_RAPSD
#         VIS_mean_RAPSD
#         VIS_std_RAPSD
#         fake_mean_RAPSD
#         fake_std_RAPSD
#
#     Args:
#         csv_path: path to mean_rapsd.csv.
#         output_dir: directory to save the figure.
#         output_name: output figure name.
#         eps: numerical stability term.
#         show_std: whether to show mean ± std shaded regions.
#         log_y: whether to plot log normalized RAPSD.
#     """
#
#     csv_path = Path(csv_path)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # sep=None 可以自动识别逗号、制表符等分隔符
#     df = pd.read_csv(csv_path, sep=None, engine="python")
#
#     # 去掉列名里可能存在的空格
#     df.columns = [c.strip() for c in df.columns]
#
#     required_columns = [
#         "normalized_frequency",
#         "IR_mean_RAPSD",
#         "IR_std_RAPSD",
#         "VIS_mean_RAPSD",
#         "VIS_std_RAPSD",
#         "fake_mean_RAPSD",
#         "fake_std_RAPSD",
#     ]
#
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing required column: {col}\n"
#                 f"Current columns are: {list(df.columns)}"
#             )
#
#     # 转成数值，防止 CSV 中存在字符串格式
#     for col in required_columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
#     df = df.dropna(subset=required_columns)
#
#     freq = df["normalized_frequency"].to_numpy()
#
#     ir_mean = df["IR_mean_RAPSD"].to_numpy()
#     ir_std = df["IR_std_RAPSD"].to_numpy()
#
#     vis_mean = df["VIS_mean_RAPSD"].to_numpy()
#     vis_std = df["VIS_std_RAPSD"].to_numpy()
#
#     fake_mean = df["fake_mean_RAPSD"].to_numpy()
#     fake_std = df["fake_std_RAPSD"].to_numpy()
#
#     plt.figure(figsize=(7, 5))
#
#     def transform_y(y):
#         if log_y:
#             return np.log(y + eps)
#         return y
#
#     def plot_one_curve(freq, mean, std, label):
#         y = transform_y(mean)
#
#         plt.plot(freq, y, label=label, linewidth=2)
#
#         if show_std:
#             lower = np.maximum(mean - std, eps)
#             upper = mean + std
#
#             y_lower = transform_y(lower)
#             y_upper = transform_y(upper)
#
#             plt.fill_between(
#                 freq,
#                 y_lower,
#                 y_upper,
#                 alpha=0.2
#             )
#
#     plot_one_curve(freq, ir_mean, ir_std, label="Infrared")
#     plot_one_curve(freq, vis_mean, vis_std, label="Visible")
#     plot_one_curve(freq, fake_mean, fake_std, label="Fake / HSLE")
#
#     plt.xlabel("Normalized radial frequency")
#
#     if log_y:
#         plt.ylabel("Log normalized RAPSD")
#         plt.title("Mean Log-RAPSD Curves")
#     else:
#         plt.ylabel("Normalized RAPSD")
#         plt.title("Mean RAPSD Curves")
#
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     fig_path = output_dir / output_name
#     plt.savefig(fig_path, dpi=300)
#     plt.close()
#
#     print(f"Figure saved to: {fig_path}")
#
#
# if __name__ == "__main__":
#     csv_path = r"D:\A_my_study\visdrone\rapsd_results\mean_rapsd.csv"
#     output_dir = r"D:\A_my_study\visdrone\rapsd_results"
#
#     plot_rapsd_from_csv(
#         csv_path=csv_path,
#         output_dir=output_dir,
#         output_name="rapsd_curve_from_csv.png",
#         show_std=True,
#         log_y=True
#     )



from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_image_gray(path):
    """
    Read image as grayscale float32.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / img.max()

    return img


def compute_rapsd(img, num_bins=128, eps=1e-8, use_hann_window=True):
    """
    Compute normalized Radial Average Power Spectral Density.
    """

    img = img.astype(np.float32)

    # Remove DC component
    img = img - np.mean(img)

    h, w = img.shape

    # Apply Hann window to reduce boundary artifacts
    if use_hann_window:
        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window = np.outer(win_y, win_x)
        img = img * window

    # FFT and power spectrum
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Radial distance map
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Normalize radius to [0, 1]
    r = r / (r.max() + eps)

    # Radial bins
    bins = np.linspace(0, 1, num_bins + 1)
    rapsd = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            rapsd[i] = power[mask].mean()

    # Normalize RAPSD as a distribution
    rapsd_norm = rapsd / (np.sum(rapsd) + eps)

    return rapsd_norm


def log_rapsd_distance(rapsd_a, rapsd_b, eps=1e-8):
    """
    Compute log-RAPSD distance between two RAPSD curves.
    """
    return np.mean(np.abs(np.log(rapsd_a + eps) - np.log(rapsd_b + eps)))


def get_image_files(folder):
    """
    Get image files in a folder.
    """
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    folder = Path(folder)

    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(files)


def match_triplets_by_stem(ir_folder, vis_folder, fake_folder):
    """
    Match infrared, visible, and fake images by filename stem.

    Example:
        infrared/0001.png
        visible/0001.png
        fake/0001.png

    These three images will be matched because their stems are all '0001'.
    """

    ir_files = get_image_files(ir_folder)
    vis_files = get_image_files(vis_folder)
    fake_files = get_image_files(fake_folder)

    ir_dict = {p.stem: p for p in ir_files}
    vis_dict = {p.stem: p for p in vis_files}
    fake_dict = {p.stem: p for p in fake_files}

    common_stems = sorted(
        set(ir_dict.keys()) &
        set(vis_dict.keys()) &
        set(fake_dict.keys())
    )

    triplets = []
    for stem in common_stems:
        triplets.append((ir_dict[stem], vis_dict[stem], fake_dict[stem]))

    print(f"Found IR images: {len(ir_files)}")
    print(f"Found VIS images: {len(vis_files)}")
    print(f"Found fake images: {len(fake_files)}")
    print(f"Matched triplets: {len(triplets)}")

    missing_ir = sorted((set(vis_dict.keys()) | set(fake_dict.keys())) - set(ir_dict.keys()))
    missing_vis = sorted((set(ir_dict.keys()) | set(fake_dict.keys())) - set(vis_dict.keys()))
    missing_fake = sorted((set(ir_dict.keys()) | set(vis_dict.keys())) - set(fake_dict.keys()))

    if len(missing_ir) > 0:
        print(f"Warning: {len(missing_ir)} images have no matched infrared image.")

    if len(missing_vis) > 0:
        print(f"Warning: {len(missing_vis)} images have no matched visible image.")

    if len(missing_fake) > 0:
        print(f"Warning: {len(missing_fake)} images have no matched fake image.")

    return triplets


def compute_dataset_rapsd(
    ir_folder,
    vis_folder,
    fake_folder,
    output_dir,
    num_bins=128,
    resize_to_ir=True,
    eps=1e-8,
    shade_type="ci"
):
    """
    Compute dataset-level RAPSD for infrared, visible, and fake images.

    Args:
        ir_folder: folder of infrared images.
        vis_folder: folder of visible images.
        fake_folder: folder of generated / enhanced / HSLE images.
        output_dir: folder to save results.
        num_bins: number of radial frequency bins.
        resize_to_ir: resize visible and fake images to infrared size.
        eps: numerical stability term.
        shade_type:
            "ci"  : plot 95% confidence interval.
            "std" : plot standard deviation.
            None  : no shaded region.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triplets = match_triplets_by_stem(ir_folder, vis_folder, fake_folder)

    if len(triplets) == 0:
        raise RuntimeError("No matched image triplets found. Please check file names.")

    ir_rapsd_list = []
    vis_rapsd_list = []
    fake_rapsd_list = []

    vis_ir_distance_list = []
    fake_ir_distance_list = []

    per_image_records = []

    for idx, (ir_path, vis_path, fake_path) in enumerate(triplets):
        ir = read_image_gray(ir_path)
        vis = read_image_gray(vis_path)
        fake = read_image_gray(fake_path)

        # Make sure all paired images have the same size
        if resize_to_ir:
            h, w = ir.shape
            vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)
            fake = cv2.resize(fake, (w, h), interpolation=cv2.INTER_AREA)
        else:
            h, w = vis.shape
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_AREA)
            fake = cv2.resize(fake, (w, h), interpolation=cv2.INTER_AREA)

        ir_rapsd = compute_rapsd(ir, num_bins=num_bins, eps=eps)
        vis_rapsd = compute_rapsd(vis, num_bins=num_bins, eps=eps)
        fake_rapsd = compute_rapsd(fake, num_bins=num_bins, eps=eps)

        dist_vis_ir = log_rapsd_distance(vis_rapsd, ir_rapsd, eps=eps)
        dist_fake_ir = log_rapsd_distance(fake_rapsd, ir_rapsd, eps=eps)

        ir_rapsd_list.append(ir_rapsd)
        vis_rapsd_list.append(vis_rapsd)
        fake_rapsd_list.append(fake_rapsd)

        vis_ir_distance_list.append(dist_vis_ir)
        fake_ir_distance_list.append(dist_fake_ir)

        record = {
            "index": idx,
            "infrared_name": ir_path.name,
            "visible_name": vis_path.name,
            "fake_name": fake_path.name,
            "log_RAPSD_distance_VIS_IR": dist_vis_ir,
            "log_RAPSD_distance_fake_IR": dist_fake_ir,
        }

        for b in range(num_bins):
            record[f"IR_bin_{b}"] = ir_rapsd[b]
            record[f"VIS_bin_{b}"] = vis_rapsd[b]
            record[f"fake_bin_{b}"] = fake_rapsd[b]

        per_image_records.append(record)

        print(
            f"[{idx + 1}/{len(triplets)}] "
            f"{ir_path.name} | {vis_path.name} | {fake_path.name} | "
            f"VIS-IR = {dist_vis_ir:.6f} | "
            f"fake-IR = {dist_fake_ir:.6f}"
        )

    ir_rapsd_array = np.stack(ir_rapsd_list, axis=0)
    vis_rapsd_array = np.stack(vis_rapsd_list, axis=0)
    fake_rapsd_array = np.stack(fake_rapsd_list, axis=0)

    vis_ir_distance_array = np.array(vis_ir_distance_list)
    fake_ir_distance_array = np.array(fake_ir_distance_list)

    # Mean and std of normalized RAPSD
    ir_mean = ir_rapsd_array.mean(axis=0)
    ir_std = ir_rapsd_array.std(axis=0)

    vis_mean = vis_rapsd_array.mean(axis=0)
    vis_std = vis_rapsd_array.std(axis=0)

    fake_mean = fake_rapsd_array.mean(axis=0)
    fake_std = fake_rapsd_array.std(axis=0)

    # Use frequency bin centers
    freq = (np.arange(num_bins) + 0.5) / num_bins

    # Save mean RAPSD
    mean_df = pd.DataFrame({
        "normalized_frequency": freq,
        "IR_mean_RAPSD": ir_mean,
        "IR_std_RAPSD": ir_std,
        "VIS_mean_RAPSD": vis_mean,
        "VIS_std_RAPSD": vis_std,
        "fake_mean_RAPSD": fake_mean,
        "fake_std_RAPSD": fake_std,
    })

    mean_csv_path = output_dir / "mean_rapsd.csv"
    mean_df.to_csv(mean_csv_path, index=False)

    # Save per-image RAPSD
    per_image_df = pd.DataFrame(per_image_records)
    per_image_csv_path = output_dir / "per_image_rapsd.csv"
    per_image_df.to_csv(per_image_csv_path, index=False)

    # Save summary
    improvement = (
        (vis_ir_distance_array.mean() - fake_ir_distance_array.mean())
        / (vis_ir_distance_array.mean() + eps)
        * 100
    )

    summary_df = pd.DataFrame({
        "metric": [
            "number_of_triplets",
            "mean_log_RAPSD_distance_VIS_IR",
            "std_log_RAPSD_distance_VIS_IR",
            "mean_log_RAPSD_distance_fake_IR",
            "std_log_RAPSD_distance_fake_IR",
            "relative_improvement_fake_over_VIS_percent",
        ],
        "value": [
            len(triplets),
            vis_ir_distance_array.mean(),
            vis_ir_distance_array.std(),
            fake_ir_distance_array.mean(),
            fake_ir_distance_array.std(),
            improvement,
        ]
    })

    summary_csv_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # =========================
    # Plot mean log-RAPSD curve
    # =========================

    # Important:
    # For visualization, compute mean/std in log-domain.
    # This avoids strange vertical spikes caused by log(mean - std).
    ir_log = np.log(ir_rapsd_array + eps)
    vis_log = np.log(vis_rapsd_array + eps)
    fake_log = np.log(fake_rapsd_array + eps)

    ir_log_mean = ir_log.mean(axis=0)
    vis_log_mean = vis_log.mean(axis=0)
    fake_log_mean = fake_log.mean(axis=0)

    ir_log_std = ir_log.std(axis=0)
    vis_log_std = vis_log.std(axis=0)
    fake_log_std = fake_log.std(axis=0)

    n = len(triplets)

    plt.figure(figsize=(7, 5))

    plt.plot(freq, ir_log_mean, label="Infrared")
    plt.plot(freq, vis_log_mean, label="Visible")
    plt.plot(freq, fake_log_mean, label="Fake / HSLE")

    if shade_type is not None:
        if shade_type == "ci":
            scale = 1.96 / np.sqrt(n)
            ir_shade = ir_log_std * scale
            vis_shade = vis_log_std * scale
            fake_shade = fake_log_std * scale
        elif shade_type == "std":
            ir_shade = ir_log_std
            vis_shade = vis_log_std
            fake_shade = fake_log_std
        else:
            raise ValueError("shade_type must be 'ci', 'std', or None.")

        plt.fill_between(
            freq,
            ir_log_mean - ir_shade,
            ir_log_mean + ir_shade,
            alpha=0.2
        )

        plt.fill_between(
            freq,
            vis_log_mean - vis_shade,
            vis_log_mean + vis_shade,
            alpha=0.2
        )

        plt.fill_between(
            freq,
            fake_log_mean - fake_shade,
            fake_log_mean + fake_shade,
            alpha=0.2
        )

    plt.xlabel("Normalized radial frequency")
    plt.ylabel("Log normalized RAPSD")
    plt.title("Mean Log-RAPSD Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = output_dir / "rapsd_mean_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("\nDone.")
    print(f"Mean RAPSD saved to: {mean_csv_path}")
    print(f"Per-image RAPSD saved to: {per_image_csv_path}")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Figure saved to: {fig_path}")

    print("\nSummary:")
    print(f"Number of triplets: {len(triplets)}")
    print(f"Mean log-RAPSD distance VIS-IR: {vis_ir_distance_array.mean():.6f}")
    print(f"Std log-RAPSD distance VIS-IR: {vis_ir_distance_array.std():.6f}")
    print(f"Mean log-RAPSD distance fake-IR: {fake_ir_distance_array.mean():.6f}")
    print(f"Std log-RAPSD distance fake-IR: {fake_ir_distance_array.std():.6f}")
    print(f"Relative improvement fake over VIS: {improvement:.2f}%")


if __name__ == "__main__":
    infrared_folder = r"D:\A_my_study\visdrone\rapsd_inf"
    visible_folder = r"D:\A_my_study\visdrone\rapsd_vis"
    fake_folder = r"D:\A_my_study\visdrone\rapsd_vis_fake"

    output_folder = r"D:\A_my_study\visdrone\rapsd_results1"

    compute_dataset_rapsd(
        ir_folder=infrared_folder,
        vis_folder=visible_folder,
        fake_folder=fake_folder,
        output_dir=output_folder,
        num_bins=128,
        resize_to_ir=True,
        shade_type="std"   # 可选："ci", "std", None
    )
