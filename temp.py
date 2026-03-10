import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ========== 你的原有模块（保持不变）==========
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
        low_col = F.conv2d(x, h_low_col, stride=(2,1), groups=C)  # [B, C, H//2, W]
        high_col = F.conv2d(x, h_high_col, stride=(2,1), groups=C)  # [B, C, H//2, W]

        # Step 2: Apply row-wise filtering (horizontal direction)
        ll = F.conv2d(low_col, h_low_row, stride=(1,2), groups=C)  # LL: low-low
        lh = F.conv2d(low_col, h_high_row, stride=(1,2), groups=C)  # LH: low-high
        hl = F.conv2d(high_col, h_low_row, stride=(1,2), groups=C)  # HL: high-low
        hh = F.conv2d(high_col, h_high_row, stride=(1,2), groups=C)  # HH: high-high

        return ll, lh, hl, hh


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ll, lh, hl, hh):
        """
        Haar IDWT using direct reconstruction formula.
        Input shapes: [B, C, H, W]
        Output shape: [B, C, 2*H, 2*W]
        """
        # Ensure all inputs have same shape
        assert ll.shape == lh.shape == hl.shape == hh.shape

        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)

        # Reconstruct four pixels from each coefficient quadruple
        out[:, :, 0::2, 0::2] = ll + lh + hl + hh
        out[:, :, 0::2, 1::2] = ll - lh + hl - hh
        out[:, :, 1::2, 0::2] = ll + lh - hl - hh
        out[:, :, 1::2, 1::2] = ll - lh - hl + hh

        return out


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
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np
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


class style_transform(nn.Module):
    def __init__(self, alpha_range=(0.8, 1.2), beta_range=(0.8, 1.2)):
        super(style_transform, self).__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.dwt = DWT()
        self.style = Style(alpha_range=alpha_range, beta_range=beta_range)
        self.idwt = IDWT()

    def forward(self, x):
        # Convert tensor to numpy array to detect content bbox
        img_np = x.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]

        batch_size = img_np.shape[0]
        transformed_imgs = []

        for i in range(batch_size):
            img_i = img_np[i]  # [H, W, C]
            y1, y2, x1, x2 = find_content_bbox(img_i)

            # Extract the content region
            roi = img_i[y1:y2, x1:x2, :]  # [h, w, C]
            roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).to(x.device)  # [1, C, h, w]

            # Apply DWT, style perturbation, and IDWT on each channel separately
            ll, lh, hl, hh = self.dwt(roi_tensor)
            # ll: [B, C, H//2, W//2]
            energy = torch.abs(ll)  # 或 ll.relu()
            # 归一化到 [0.0001, 0.01]
            attenuation = 0.01 * torch.exp(-energy / 50.0)  # 亮区衰减更强
            ll_perturbed = self.style(ll)
            lh = attenuation*lh
            hh = attenuation*hh
            hl = attenuation*hl
            roi_recon = self.idwt(ll_perturbed, lh, hl, hh)

            # Resize reconstructed ROI to original size
            roi_recon_np = roi_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [h, w, C]
            recon_img = img_i.copy()
            recon_img[y1:y2, x1:x2, :] = roi_recon_np

            transformed_imgs.append(recon_img)

        # Combine transformed images back into a single tensor
        transformed_imgs = np.stack(transformed_imgs, axis=0)  # [B, H, W, C]
        transformed_imgs = torch.from_numpy(transformed_imgs).permute(0, 3, 1, 2).to(x.device)  # [B, C, H, W]

        return transformed_imgs
# ========== 批量处理文件夹 ==========
import torchvision.transforms as transforms

# 假设 style_transform 已经定义好
transformer = style_transform(alpha_range=(0.8, 1.2), beta_range=(0.8, 1.2))


# 加载并预处理图像
def load_image(image_path, device='cpu'):
    # 定义图像转换方式
    img_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并且数值范围变为[0, 1]
    ])

    # 使用OpenCV读取图像
    img = cv2.imread(image_path)
    # 将图像从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 应用转换
    img_tensor = img_transform(img)
    # 添加batch维度 [C, H, W] -> [B, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor, img.shape[:2]  # 返回原始图像尺寸（高度，宽度）


# 图像路径
image_path = 'D:/A_my_study/visdrone/train/daytime/images_rgb1/00213.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载图像
img_tensor, original_shape = load_image(image_path, device=device)

# 确保模型在正确的设备上
transformer.to(device)

# 使用模型变换图像
transformed_img_tensor = transformer(img_tensor)

# 移动到CPU并转换为numpy数组以显示或保存
transformed_img = transformed_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
transformed_img = np.clip(transformed_img, 0, 1)  # 确保像素值在[0, 1]之间

# 如果需要，可以将图像调整回原始大小
if transformed_img.shape[:2] != original_shape:
    transformed_img = cv2.resize(transformed_img, (original_shape[1], original_shape[0]))

# 显示结果（这里假设你在一个支持matplotlib的环境中）


plt.imshow(transformed_img)
plt.title("Transformed Image")
plt.axis('off')
plt.show()

# 或者如果你想保存结果
output_path = "D:/A_my_study/visdrone/train/daytime/path_to_save_transformed_image.jpg"
plt.imsave(output_path, transformed_img)