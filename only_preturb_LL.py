import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image
import torch


# ----------------------------
# Normalization Perturbation (NP)
# ----------------------------
def normalization_perturbation(ll, alpha_range=(0.8, 1.2), beta_range=(0.8, 1.2)):
    """
    对 LL 子带应用 NP 扰动：
        y = σ*_s * (x - μ) / σ + μ*_s
        σ*_s = α * σ,   μ*_s = β * μ
    """
    # 计算均值和标准差（沿空间维度）
    mu = np.mean(ll)
    sigma = np.std(ll) + 1e-8  # 防除零

    # 随机采样 α 和 β
    alpha = np.random.uniform(*alpha_range)
    beta = np.random.uniform(*beta_range)

    # 扰动后的统计量
    sigma_star = alpha * sigma
    mu_star = beta * mu

    # 应用 NP
    ll_norm = (ll - mu) / sigma
    ll_perturbed = sigma_star * ll_norm + mu_star

    return ll_perturbed


def find_content_bbox(img, white_threshold=250):
    """
    自动检测非白边区域的 bounding box
    返回 (y1, y2, x1, x2)
    """
    # 转为灰度判断（更鲁棒）
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    # 找到非白区域（< white_threshold）
    mask = gray < white_threshold

    if not np.any(mask):
        # 全是白边？返回全图
        H, W = gray.shape
        return 0, H, 0, W

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # 可选：加一点 padding 防止切太紧
    pad = 2
    y1 = max(0, y1 - pad)
    y2 = min(gray.shape[0], y2 + pad + 1)
    x1 = max(0, x1 - pad)
    x2 = min(gray.shape[1], x2 + pad + 1)

    return y1, y2, x1, x2


# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    # 1. 读取图像（替换为你自己的路径）
    image_path = "D:/A_my_study/minyong/vis/all/images/DJI_20260416151632_0001_V_00091.jpg"  # ←← 修改这里！
    # DJI_20260416151632_0001_V_00193
    # DJI_20260416151632_0001_V_00091
    img = Image.open(image_path)  # 保持RGB格式
    img_np = np.array(img, dtype=np.float32)

    # 初始化重构图像（复制原图，白边保留）
    img_recon = img_np.copy()

    # 2. 检测有效内容区域（排除白边）
    y1, y2, x1, x2 = find_content_bbox(img_np, white_threshold=250)
    print(f"Content ROI: [{y1}:{y2}, {x1}:{x2}]")

    # 3. 对每个通道分别进行 DWT 和 NP 扰动
    for channel in range(3):  # R, G, B
        # 提取单个通道的有效区域
        img_channel = img_np[y1:y2, x1:x2, channel]

        # 4. DWT 分解（一级）
        coeffs = pywt.dwt2(img_channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        energy = np.abs(LL)
        attenuation = 0.01 * np.exp(-energy / 50.0)  # 亮区衰减更强
        HH = HH * attenuation
        LH = LH * attenuation
        HL = HL * attenuation

        # 5. 对 LL 应用 NP 扰动
        LL_perturbed = normalization_perturbation(
            LL,
            alpha_range=(0.7, 1.3),
            beta_range=(0.7, 1.3)
        )

        # 6. 用扰动后的 LL 重构图像
        coeffs_perturbed = (LL_perturbed, (LH, HL, HH))
        img_channel_recon = pywt.idwt2(coeffs_perturbed, 'haar')

        # 确保重构图尺寸与原图一致（dwt2/idwt2 可能因奇偶尺寸有1像素差异）
        img_channel_recon = img_channel_recon[:img_channel.shape[0], :img_channel.shape[1]]

        # 将重构后的通道放回原图的对应位置
        img_recon[y1:y2, x1:x2, channel] = img_channel_recon

    # 7. 可视化对比
    plt.figure(figsize=(15, 5))

    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img_np.astype(np.uint8))
    plt.title("Original Image")
    plt.axis('off')

    # 重构图（NP 扰动后）
    plt.subplot(1, 3, 2)
    plt.imshow(img_recon.astype(np.uint8))
    plt.title("Reconstructed (LL perturbed by NP)")
    plt.axis('off')

    # 差异图
    diff = np.abs(img_np - img_recon)
    diff_mean = np.mean(diff, axis=2)  # 计算平均差异
    plt.subplot(1, 3, 3)
    plt.imshow(diff_mean, cmap='hot')
    plt.title("Absolute Difference")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("dwt_np_reconstruction_rgb_no_white_border.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    mse = np.mean((img_np - img_recon) ** 2)
    mse_content = np.mean((img_np[y1:y2, x1:x2] - img_recon[y1:y2, x1:x2]) ** 2)
    print(f"Reconstruction MSE after NP on entire image: {mse:.4f}")
    print(f"MSE on content region: {mse_content:.4f}")