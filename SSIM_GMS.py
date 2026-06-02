import cv2
import os
import glob
import csv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# ----------------------------
# 核心指标计算函数
# ----------------------------
def calculate_ssim_score(img1_bgr, img2_bgr):
    """
    计算两幅图像的 SSIM 分数 (接收已读取的 numpy 数组)
    """
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        # 如果尺寸有微小差异，强制 resize 对齐
        gray2 = cv2.resize(gray2, (gray1.shape, gray1.shape))

    score = ssim(gray1, gray2, data_range=255)
    return score


def calculate_gms_similarity(img1_bgr, img2_bgr, C=1e-3):
    """
    计算两幅图像的梯度幅值相似性 (Gradient Magnitude Similarity)
    """

    def get_gradient_magnitude(img_bgr):
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img_bgr.astype(np.float64)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return magnitude

    # 确保尺寸一致
    if img1_bgr.shape[:2] != img2_bgr.shape[:2]:
        img2_bgr = cv2.resize(img2_bgr,
                              (img1_bgr.shape, img1_bgr.shape))

    M1 = get_gradient_magnitude(img1_bgr)
    M2 = get_gradient_magnitude(img2_bgr)

    numerator = 2 * M1 * M2 + C
    denominator = M1 ** 2 + M2 ** 2 + C
    gms_map = numerator / denominator

    return np.mean(gms_map)


def find_fake_image(fake_folder, original_name):
    """
    根据原图文件名寻找对应的 fake 图像。
    例如 original_name = 00017，则查找 00017_fake.jpg/png/bmp/jpeg。
    """
    extensions = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    for ext in extensions:
        # 尝试小写后缀
        fake_path = os.path.join(fake_folder, f"{original_name}_fake.{ext}")
        if os.path.exists(fake_path):
            return fake_path
        # 尝试大写后缀
        fake_path_upper = os.path.join(fake_folder, f"{original_name}_fake.{ext.upper()}")
        if os.path.exists(fake_path_upper):
            return fake_path_upper
    return None


# ----------------------------
# 批量处理主流程
# ----------------------------
def batch_calculate_metrics(original_folder, fake_folder, save_csv_path="metrics_results.csv"):
    # 1. 获取原图文件夹中所有图片路径
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
                        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF"]
    original_paths = []
    for ext in image_extensions:
        original_paths.extend(glob.glob(os.path.join(original_folder, ext)))

    # 去重并排序，保证处理顺序稳定
    original_paths = sorted(list(set(original_paths)))

    if len(original_paths) == 0:
        print(f"原图文件夹中未找到图像: {original_folder}")
        return

    results = []
    skipped = []

    print(f"共找到 {len(original_paths)} 张原始图像，开始批量计算 SSIM 与 S_grad...")

    # 2. 带进度条的批量循环
    for original_path in tqdm(original_paths, desc="Calculating Metrics"):
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)

        # 寻找配对的 fake 图
        fake_path = find_fake_image(fake_folder, name)

        if fake_path is None:
            skipped.append((filename, "未找到对应 fake 图像"))
            continue

        try:
            # 读取图像 (只读取一次，同时传给两个指标函数)
            img_orig = cv2.imread(original_path)
            img_fake = cv2.imread(fake_path)

            if img_orig is None or img_fake is None:
                raise ValueError("图像读取失败")

            # 计算指标
            ssim_val = calculate_ssim_score(img_orig, img_fake)
            gms_val = calculate_gms_similarity(img_orig, img_fake)

            results.append({
                "original_image": filename,
                "fake_image": os.path.basename(fake_path),
                "ssim": ssim_val,
                "s_grad": gms_val
            })

        except Exception as e:
            skipped.append((filename, str(e)))
            continue

    if len(results) == 0:
        print("没有成功计算任何图像对的指标。")
        return

    # 3. 统计计算
    ssim_values = np.array([item["ssim"] for item in results])
    gms_values = np.array([item["s_grad"] for item in results])

    mean_ssim, std_ssim = np.mean(ssim_values), np.std(ssim_values)
    mean_gms, std_gms = np.mean(gms_values), np.std(gms_values)

    # 4. 终端输出最终结果
    print("\n" + "=" * 50)
    print(f"批量计算完成！共成功处理 {len(results)} 张图片，跳过 {len(skipped)} 张。")
    print("-" * 50)
    print(f"【SSIM 统计结果】")
    print(f"  平均值 (Mean): {mean_ssim:.6f}")
    print(f"  标准差 (Std):  {std_ssim:.6f}")
    print("-" * 50)
    print(f"【S_grad (GMS) 统计结果】")
    print(f"  平均值 (Mean): {mean_gms:.6f}")
    print(f"  标准差 (Std):  {std_gms:.6f}")
    print("=" * 50)

    # 简单的评价参考
    if mean_ssim > 0.85 and mean_gms > 0.70:
        print("结论：语义结构保持极佳，且边缘衰减符合红外模拟预期。")
    elif mean_ssim < 0.70:
        print("警告：SSIM 过低，可能存在严重的语义信息丢失！")
    elif mean_gms < 0.50:
        print("警告：S_grad 过低，边缘强度损失严重，可能导致目标检测漏检！")

    # 5. 保存 CSV (方便后续画散点图)
    with open(save_csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["original_image", "fake_image", "ssim", "s_grad"])
        for item in results:
            writer.writerow([
                item["original_image"],
                item["fake_image"],
                f"{item['ssim']:.6f}",
                f"{item['s_grad']:.6f}"
            ])
    print(f"\n详细数据已保存到: {save_csv_path} (可直接用于绘制 SSIM-S_grad 散点图)")

    if skipped:
        print("\n以下图像被跳过:")
        for item in skipped[:10]:
            print(f"  {item} -> {item}")
        if len(skipped) > 10:
            print(f"  ... 还有 {len(skipped) - 10} 个未显示")


if __name__ == "__main__":
    # ================= 配置区域 =================
    original_folder = "D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/vis"  # 原图文件夹
    fake_folder = "D:/Deeplearning_code/yolov8/ultralytics/HSLE_for_effectiveness/output_fake"  # _fake图文件夹
    save_csv_path = "metrics_results.csv"  # 结果保存路径
    # ===========================================

    batch_calculate_metrics(original_folder, fake_folder, save_csv_path)