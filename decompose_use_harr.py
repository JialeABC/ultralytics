import torch
import torch.nn.functional as f
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class DWT(nn.Module):
    def __init__(self, wavelet='haar', requires_grad=False):
        """
        2D Discrete Wavelet Transform (single-level).
        Outputs: LL, LH, HL, HH (each same spatial size as input // 2)

        Args:
            wavelet: currently only 'haar' is implemented.
            requires_grad: if True, filters are trainable (for learnable DWT).
        """
        super().__init__()
        self.wavelet = wavelet
        if wavelet == 'haar':
            # Haar low-pass and high-pass filters
            h_low = torch.tensor([1., 1.]) / 2.0  # Low-pass (scaling)
            h_high = torch.tensor([1., -1.]) / 2.0  # High-pass (wavelet)
        else:
            raise NotImplementedError("Only 'haar' wavelet is supported.")

        # Register filters as buffers (non-trainable by default)
        self.register_buffer('h_low', h_low[None, None, :, None])  # [1, 1, 2, 1] for vertical conv
        self.register_buffer('h_high', h_high[None, None, :, None])  # [1, 1, 2, 1]

        if requires_grad:
            self.h_low = nn.Parameter(self.h_low)
            self.h_high = nn.Parameter(self.h_high)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [B, C, H, W]
        Returns:
            ll, lh, hl, hh: each of shape [B, C, H//2, W//2]
        """
        assert x.dim() == 4, "Input must be 4D tensor [B, C, H, W]"
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even."

        # Expand filters to match input channels
        h_low_col = self.h_low.expand(C, -1, -1, -1)  # [C, 1, 2, 1]
        h_high_col = self.h_high.expand(C, -1, -1, -1)  # [C, 1, 2, 1]
        h_low_row = self.h_low.transpose(2, 3).expand(C, -1, -1, -1)  # [C, 1, 1, 2]
        h_high_row = self.h_high.transpose(2, 3).expand(C, -1, -1, -1)  # [C, 1, 1, 2]

        # Step 1: Apply column-wise filtering (vertical direction)
        # Use groups=C to apply per-channel convolution
        low_col = f.conv2d(x, h_low_col, stride=(2,1), groups=C)  # [B, C, H//2, W]
        high_col = f.conv2d(x, h_high_col, stride=(2,1), groups=C)  # [B, C, H//2, W]

        # Step 2: Apply row-wise filtering (horizontal direction)
        ll = f.conv2d(low_col, h_low_row, stride=(1,2), groups=C)  # LL: low-low
        lh = f.conv2d(low_col, h_high_row, stride=(1,2), groups=C)  # LH: low-high
        hl = f.conv2d(high_col, h_low_row, stride=(1,2), groups=C)  # HL: high-low
        hh = f.conv2d(high_col, h_high_row, stride=(1,2), groups=C)  # HH: high-high

        return ll, lh, hl, hh

class Style(nn.Module):
    def __init__(self, alpha_range=(0.8, 1.2), beta_range=(0.8, 1.2)):
        super().__init__()
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def forward(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3], keepdim=True) + 1e-6

        alpha = torch.empty_like(mu).uniform_(*self.alpha_range)
        beta = torch.empty_like(mu).uniform_(*self.beta_range)

        sigma_star = alpha * sigma
        mu_star = beta * mu

        x_norm = (x - mu) / sigma
        x_perturbed = x_norm * sigma_star + mu_star

        return x_perturbed

def find_content_bbox(img_np, white_threshold=250):
    """
    自动检测非白边区域的 bounding box
    返回 (y1, y2, x1, x2)
    """
    # 转为灰度判断（更鲁棒）
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np

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
#==========================================以下仅仅是分解==============================================#
# def load_image(image_path, transform=None):
#     """Load and preprocess the image."""
#     img = Image.open(image_path) #.convert('L')  # Convert to grayscale
#     if transform:
#         img = transform(img)
#     return img.unsqueeze(0)  # Add batch dimension
#
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize for simplicity
#     transforms.ToTensor(),
# ])
#
# image_path = 'D:/A_my_study/visdrone/train/daytime/images_rgb1/00233.jpg'  # 替换成你的图片路径
# img_tensor = load_image(image_path, transform=transform)
#
# # Initialize DWT with Haar wavelet
# dwt = DWT(wavelet='haar')
#
# # Apply DWT on the input image tensor
# ll, lh, hl, hh = dwt(img_tensor)
#
# # Convert tensors back to images for visualization
# to_pil = transforms.ToPILImage()
#
# fig, ax = plt.subplots(2, 2, figsize=(8, 8))
#
# ax[0, 0].imshow(to_pil(img_tensor.squeeze().cpu()), cmap='gray')
# ax[0, 0].set_title('Original Image')
# ax[0, 0].axis('off')
#
# ax[0, 1].imshow(to_pil(ll.squeeze().cpu()), cmap='gray')
# ax[0, 1].set_title('LL (Approximation)')
# ax[0, 1].axis('off')
#
# ax[1, 0].imshow(to_pil(lh.squeeze().cpu()), cmap='gray')
# ax[1, 0].set_title('LH (Horizontal Detail)')
# ax[1, 0].axis('off')
#
# ax[1, 1].imshow(to_pil(hl.squeeze().cpu()), cmap='gray')
# ax[1, 1].set_title('HL (Vertical Detail)')
# ax[1, 1].axis('off')
#
# plt.figure(figsize=(5, 5))
# plt.imshow(to_pil(hh.squeeze().cpu()), cmap='gray')
# plt.title('HH (Diagonal Detail)')
# plt.axis('off')
#
# plt.show()

#======================================对高频分量HH,HL,LH,LL抑制===========================================#
# def load_image(image_path, transform=None):
#     """Load and preprocess the image."""
#     img = Image.open(image_path)#.convert('L')  # Convert to grayscale
#     if transform:
#         img = transform(img)
#     return img.unsqueeze(0)  # Add batch dimension
#
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize for simplicity
#     transforms.ToTensor(),
# ])
#
# image_path = 'D:/A_my_study/visdrone/train/daytime/images_rgb1/00213.jpg'  # 替换成你的图片路径
# img_tensor = load_image(image_path, transform=transform)
#
# # Initialize DWT with Haar wavelet
# dwt = DWT(wavelet='haar')
#
# # Apply DWT on the input image tensor
# ll, lh, hl, hh = dwt(img_tensor)
#
# # 抑制高频分量
# energy = torch.abs(ll)
# attenuation = 0.01 * torch.exp(-energy / 50.0)
# hh_atten = hh * attenuation
# lh_atten = lh * attenuation
# hl_atten = hl * attenuation
#
# # Convert tensors back to images for visualization
# to_pil = transforms.ToPILImage()
#
# fig, axes = plt.subplots(3, 2, figsize=(10, 15))
#
# titles = ['Original HL', 'Attenuated HL', 'Original LH', 'Attenuated LH', 'Original HH', 'Attenuated HH']
# images = [hl, hl_atten, lh, lh_atten, hh, hh_atten]
#
# for ax, title, img in zip(axes.flatten(), titles, images):
#     ax.imshow(to_pil(img.squeeze().cpu()), cmap='gray')
#     ax.set_title(title)
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()
#================================================扰动LL前后对比图========================================================#

def perturb_ll_in_content_region(ll_tensor, white_threshold=250):
    """
    对 LL 分量仅在有效内容区域做 Style 扰动。
    Args:
        ll_tensor: [B, C, H, W] 的 LL 分量（torch.Tensor）
    Returns:
        ll_perturbed: 扰动后的 LL
    """
    device = ll_tensor.device
    B, C, H, W = ll_tensor.shape
    ll_np = ll_tensor.detach().cpu().numpy()

    # 假设 batch=1，且取第一个通道用于 bbox 检测（或取均值）
    # 若多通道，可对通道取平均作为灰度图
    if C > 1:
        gray_ll = np.mean(ll_np[0], axis=0)  # [H, W]
    else:
        gray_ll = ll_np[0, 0]

    # 归一化到 [0, 255] 以便使用 white_threshold
    gray_ll_vis = (gray_ll - gray_ll.min()) / (gray_ll.max() - gray_ll.min() + 1e-8) * 255.0
    y1, y2, x1, x2 = find_content_bbox(gray_ll_vis, white_threshold=white_threshold)

    # 创建扰动模块
    style_module = Style().to(device)

    # 克隆原始 LL 用于修改
    ll_perturbed = ll_tensor.clone()

    # 仅对有效区域应用扰动
    region = ll_tensor[:, :, y1:y2, x1:x2]
    if region.numel() > 0:
        region_perturbed = style_module(region)
        ll_perturbed[:, :, y1:y2, x1:x2] = region_perturbed

    return ll_perturbed, (y1, y2, x1, x2)


# ------------------ 可视化函数 ------------------
def visualize_ll_before_after(original_ll, perturbed_ll, bbox=None, channel_idx=0):
    """
    可视化原始 LL 和扰动后 LL（支持单张图像，batch=1）
    """
    original = original_ll[0, channel_idx].detach().cpu().numpy()
    perturbed = perturbed_ll[0, channel_idx].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original LL')
    axes[0].axis('off')

    axes[1].imshow(perturbed, cmap='gray')
    axes[1].set_title('Perturbed LL (content region only)')
    axes[1].axis('off')

    if bbox is not None:
        y1, y2, x1, x2 = bbox
        rect1 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
        rect2 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect1)
        axes[1].add_patch(rect2)

    plt.tight_layout()
    plt.show()


# ------------------ 示例使用 ------------------
if __name__ == "__main__":
    # 模拟一张带白边的图像（例如 256x256，中间 200x200 是内容）
    H, W = 256, 256
    content = np.random.rand(200, 200) * 200 + 30  # 非白内容 [30, 230]
    img = np.ones((H, W)) * 255  # 白背景
    img[28:228, 28:228] = content  # 居中放置内容

    # 转为 tensor [1, 1, H, W]
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    # DWT 分解
    dwt = DWT()
    ll, lh, hl, hh = dwt(x)

    # 仅对有效区域扰动 LL
    ll_perturbed, bbox = perturb_ll_in_content_region(ll, white_threshold=240)

    # 可视化
    visualize_ll_before_after(ll, ll_perturbed, bbox=bbox)