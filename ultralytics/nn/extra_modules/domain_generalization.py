import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as f
import kornia
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from ultralytics.nn.modules.conv import Conv2


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
        sigma = x.std(dim=[2, 3], keepdim=True) + 1e-6  # avoid div by zero

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


class FeatureStyleMix(nn.Module):
    """
    Feature-level style mixing via interpolating instance normalization statistics.
    Inspired by MixStyle (ICLR 2021) and the provided formulation.

    Args:
        p (float): Probability of applying style mixing during forward pass. Default: 0.5
        alpha (float): Parameter for Beta distribution (λ ～ Beta(α, α)). Default: 0.1
    """

    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        """
        x: [B, C, H, W] feature tensor
        returns: [B, C, H, W] mixed feature tensor (or original if not applied)
        """
        if not self.training or torch.rand(1).item() > self.p:
            return x  # 不应用混合（测试阶段或随机跳过）

        B, C, H, W = x.shape

        # Step 1: 计算每个样本的 μ 和 σ (instance-wise)
        # [B, C, 1, 1]
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3], unbiased=False, keepdim=True)

        # Step 2: 随机打乱 batch 维度，得到 f' 的统计量
        # 生成打乱索引
        shuffled_idx = torch.randperm(B, device=x.device)
        mu_prime = mu[shuffled_idx]  # [B, C, 1, 1]
        sigma_prime = sigma[shuffled_idx]

        # Step 3: 从 Beta 分布采样 λ ∈ [0,1]
        # 使用对称 Beta(α, α)，α 越小，λ 越靠近 0 或 1（极端混合）
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)

        # Step 4: 计算混合统计量 β_mix 和 γ_mix
        beta_mix = lam * mu + (1 - lam) * mu_prime  # [B, C, 1, 1]
        gamma_mix = lam * sigma + (1 - lam) * sigma_prime  # [B, C, 1, 1]

        # Step 5: 对 x 做归一化后再用混合统计量重缩放和平移
        # 先 Instance Normalization (without learnable affine)
        x_norm = (x - mu) / (sigma + 1e-6)

        # 应用混合风格
        x_mixed = gamma_mix * x_norm + beta_mix

        return x_mixed

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
            # energy = torch.abs(ll)  # 或 ll.relu()
            # # # 归一化到 [0.0001, 0.01]
            # attenuation = 0.01 * torch.exp(-energy / 50.0)  # 亮区衰减更强
            ll_perturbed = self.style(ll)
            # hh = hh * attenuation
            # lh = lh * attenuation
            # hl = hl * attenuation
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
#===========================================================上面是风格变换模块==================================================================================#

class base_disen_feature_extractor(nn.Module):
    def __init__(self,in_channels=256):
        super().__init__()
        # 使用 3x3 卷积，padding=1 保证 spatial 尺寸不变
        # 所有层保持通道数 = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        # 可选：加入非线性激活（如 ReLU）和归一化（如 BatchNorm）
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class decouple_feature(nn.Module):
    def __init__(self):
        super(decouple_feature, self).__init__()
        self.inv_extractor1 = base_disen_feature_extractor(in_channels=32)
        self.inv_extractor2 = base_disen_feature_extractor(in_channels=64)
        self.inv_extractor3 = base_disen_feature_extractor(in_channels=128)
        self.var_extractor1_rgb = base_disen_feature_extractor(in_channels=32)
        self.var_extractor1_inf = base_disen_feature_extractor(in_channels=32)
        self.var_extractor2_rgb = base_disen_feature_extractor(in_channels=64)
        self.var_extractor2_inf = base_disen_feature_extractor(in_channels=64)
        self.var_extractor3_rgb = base_disen_feature_extractor(in_channels=128)
        self.var_extractor3_inf = base_disen_feature_extractor(in_channels=128)

    def forward(self, x, x_aug):
        B,C,H,W = x.shape
        if C==32:
            inv_feature_rgb = self.inv_extractor1(x)
            var_feature_rgb = self.var_extractor1_rgb(x)
            if x_aug is not None:
                var_feature_inf = self.var_extractor1_inf(x_aug)
                inv_feature_inf = self.inv_extractor1(x_aug)
        elif C==64:
            inv_feature_rgb = self.inv_extractor2(x)
            var_feature_rgb = self.var_extractor2_rgb(x)
            if x_aug is not None:
                var_feature_inf = self.var_extractor2_inf(x_aug)
                inv_feature_inf = self.inv_extractor2(x_aug)
        else:
            inv_feature_rgb = self.inv_extractor3(x)
            var_feature_rgb = self.var_extractor3_rgb(x)
            if x_aug is not None:
                var_feature_inf = self.var_extractor3_inf(x_aug)
                inv_feature_inf = self.inv_extractor3(x_aug)
        return inv_feature_rgb, inv_feature_inf, var_feature_rgb, var_feature_inf
#===========================================================上面是特征解耦器模块==================================================================================#

class StyleAgnosticProjection(nn.Module):
    def __init__(self, dim, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        # Es: style encoder (linear + norm)
        self.style_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

class domain_agnostic(nn.Module):
    def __init__(self, channel, alpha=1.0):
        super(domain_agnostic, self).__init__()
        self.channel = channel
        self.alpha = alpha  # 训练时通常设为 1.0

        # Style encoder: linear layer + normalization (as described in your paper)
        self.style_encoder = nn.Sequential(
            nn.Linear(channel, channel),
            nn.LayerNorm(channel)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: Compute style statistics (μ_s, σ_s) per sample & channel
        # Shape: [B, C]
        mu = x.mean(dim=[2, 3])  # [B, C]
        sigma = x.std(dim=[2, 3], unbiased=False)  # [B, C]

        # Step 2: Construct style vector z_s = mu + sigma
        z_s = mu + sigma  # [B, C]

        # Step 3: Encode to style embedding s
        s = self.style_encoder(z_s)  # [B, C]

        # Step 4: Flatten x to Q: [B, N, C], N = H*W
        Q = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Step 5: Compute projection of Q onto span{s}
        # s: [B, C] -> unsqueeze to [B, 1, C] for batch matmul
        s_unsq = s.unsqueeze(1)  # [B, 1, C]

        # Compute Q * s^T -> [B, N, 1]
        qs_dot = torch.bmm(Q, s_unsq.transpose(1, 2))  # [B, N, 1]

        # Compute ||s||^2 -> [B, 1, 1]
        s_norm_sq = torch.bmm(s_unsq, s_unsq.transpose(1, 2))  # [B, 1, 1]

        # Avoid division by zero
        s_norm_sq = s_norm_sq.clamp(min=1e-8)

        # Projection coefficient: (Q s^T) / ||s||^2 -> [B, N, 1]
        coeff = qs_dot / s_norm_sq  # [B, N, 1]

        # Projected component: coeff * s -> [B, N, C]
        proj_L_Q = coeff * s_unsq  # broadcasting: [B, N, 1] * [B, 1, C] -> [B, N, C]

        # Step 6: Remove style component
        Q_hat = Q - self.alpha * proj_L_Q  # [B, N, C]

        # Step 7: Reshape back to [B, C, H, W]
        x_out = Q_hat.transpose(1, 2).view(B, C, H, W)

        return x_out

        pass

#===========================================================上面是Domain Agnostic模块==================================================================================#

class ConstrainedY2IR(nn.Module):
    def __init__(self, gamma_range=(0.6,1.5),alpha_range=(0.7,1.3),sigma_range=(0.,1.6)):
        super(ConstrainedY2IR, self).__init__()
        self.gamma_range = gamma_range
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

        #三参数预测头：全局池化 + 2层FC
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(3, 8),
                                nn.ReLU(inplace=True),
                                nn.Linear(8, 3),
                                nn.Sigmoid()
                                )
    def forward(self, x):
        p = self.fc(x)
        y = self._perturb(x, p)
        # # 转为 numpy
        # x_np = x.squeeze().permute(1, 2, 0).numpy()
        # x3_np = y.squeeze().permute(1, 2, 0).numpy()
        #
        # # 可视化
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(x_np)
        # plt.title("Original")
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(x3_np)
        # plt.title("x3 (after gamma + contrast + blur)")
        # plt.axis('off')
        # plt.show()

        return y

    def _perturb(self, x, p):
        #映射到物理空间
        gamma = self.gamma_range[0] +(self.gamma_range[1]-self.gamma_range[0])*p[:,0:1]
        alpha = self.alpha_range[0] +(self.alpha_range[1]-self.alpha_range[0])*p[:,1:2]
        sigma = self.sigma_range[0] +(self.sigma_range[1]-self.sigma_range[0])*p[:,2:3]

        #gamma处理
        x1 = x ** gamma.view(-1,1,1,1)

        #对比度拉伸
        x2 = (x1-0.5)*alpha.view(-1,1,1,1)+0.5
        x2 = torch.clamp(x2,0.,1)

        #高斯模糊
        if sigma.max().item()>0.01:
            sigma = sigma.repeat(1, 2)  # 变成 (16, 2)，两列值相同
            x3 = kornia.filters.gaussian_blur2d(x2,kernel_size=7, sigma=sigma)
        else:
            x3 = x2
        return x3
#===========================================================上面是模拟红外模块==================================================================================#

class base_FDM(nn.Module):
    def __init__(self,ch, num_classes=5):
        super(base_FDM, self).__init__()

        # 1. 特征解耦模块
        # 只为当前通道 ch 创建三个解耦模块
        self.feature_disentangle_inv = base_disen_feature_extractor(in_channels=ch)
        self.feature_disentangle_var_rgb = base_disen_feature_extractor(in_channels=ch)
        self.feature_disentangle_var_inf = base_disen_feature_extractor(in_channels=ch)

        # 2. 熵预测头（统一用 ModuleDict）
        self.entropy_head = EntropyHead(in_channels=ch, num_classes=num_classes)

class FDM(nn.Module):
    def __init__(self,inchannels,outchannels, num_classes=5):
        super(FDM, self).__init__()
        # if inchannels==32:
        self.decouple_extractor = base_FDM(ch=inchannels, num_classes=num_classes)
        # elif inchannels==64:
        #     self.decouple_extractor2 = base_FDM(ch=64, num_classes=num_classes)
        # elif inchannels==128:
        #     self.decouple_extractor3 = base_FDM(ch=128, num_classes=num_classes)
        # else:
        #     self.decouple_extractor4 = base_FDM(ch=256, num_classes=num_classes)

        self.current_mode = "RGB"  # 默认


    def forward(self, x):

        if self.current_mode == "RGB":
            F_inv = self.decouple_extractor.feature_disentangle_inv(x)
            F_var = self.decouple_extractor.feature_disentangle_var_rgb(x)
        else:
            F_inv = self.decouple_extractor.feature_disentangle_inv(x)
            F_var = self.decouple_extractor.feature_disentangle_var_inf(x)


        return F_inv, F_var


def compute_triple_loss(F_inv_rgb, F_inv_inf, F_var_rgb, F_var_inf, margin=0.5):
    """
    使用全局平均池化（GAP）将 (B, C, H, W) → (B, C)
    避免维度爆炸，同时保留通道语义。
    """
    if F_inv_rgb.dim() == 4:
        # 全局平均池化: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        F_inv_rgb = F_inv_rgb.mean(dim=[2, 3])  # 等价于 GAP
        F_inv_inf = F_inv_inf.mean(dim=[2, 3])
        F_var_rgb = F_var_rgb.mean(dim=[2, 3])
        F_var_inf = F_var_inf.mean(dim=[2, 3])

    # L2 归一化（现在是 (B, C)，C=256）
    F_inv_rgb = f.normalize(F_inv_rgb, p=2, dim=1)
    F_inv_inf = f.normalize(F_inv_inf, p=2, dim=1)
    F_var_rgb = f.normalize(F_var_rgb, p=2, dim=1)
    F_var_inf = f.normalize(F_var_inf, p=2, dim=1)

    def l2_dist(a, b):
        return torch.sum((a - b) ** 2, dim=1)

    # 三元组损失（现在在 256 维空间计算，合理！）
    loss1 = torch.relu(l2_dist(F_inv_rgb, F_inv_inf) - l2_dist(F_inv_rgb, F_var_rgb) + margin)
    loss2 = torch.relu(l2_dist(F_inv_inf, F_inv_rgb) - l2_dist(F_inv_inf, F_var_inf) + margin)

    return (loss1 + loss2).mean()

# ----------------------------
# 1. 轻量级熵预测头（固定输入通道）
# ----------------------------
class EntropyHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, C, H, W) → (N, C, 1, 1)
            nn.Flatten(),  # → (N, C)
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (N, C, H, W) or (N, C)
        """
        return self.head(x)


# ----------------------------
# 2. 熵计算函数
# ----------------------------
def compute_entropy(prob):
    """
    prob: (N, C) — 已归一化的概率分布
    returns: (N,) — 每个样本的熵
    """
    prob = torch.clamp(prob, min=1e-8, max=1.0)
    return -(prob * torch.log(prob)).sum(dim=1)


# ----------------------------
# 3. 熵对比损失函数（核心）
# ----------------------------
def compute_entropy_loss(
        F_inv_rgb,
        F_inv_inf,
        F_var_rgb,
        F_var_inf,
        entropy_head
):
    """
    计算熵对比损失：
      - 鼓励 F_inv（不变特征）预测低熵（确定）
      - 鼓励 F_var（可变特征）预测高熵（不确定）

    Args:
        F_inv_rgb, F_inv_inf, F_var_rgb, F_var_inf: (N, C, H, W) 或 (N, C)
        entropy_head: EntropyHead 实例（必须与输入通道数匹配）

    Returns:
        loss_eco: scalar tensor
    """
    # 通过同一个 head 获取 logits
    logits_inv_rgb = entropy_head(F_inv_rgb)
    logits_inv_inf = entropy_head(F_inv_inf)
    logits_var_rgb = entropy_head(F_var_rgb)
    logits_var_inf = entropy_head(F_var_inf)

    # 转为概率（使用 softmax，适用于互斥类别）
    prob_inv_rgb = f.softmax(logits_inv_rgb, dim=1)
    prob_inv_inf = f.softmax(logits_inv_inf, dim=1)
    prob_var_rgb = f.softmax(logits_var_rgb, dim=1)
    prob_var_inf = f.softmax(logits_var_inf, dim=1)

    # 计算熵
    H_inv_rgb = compute_entropy(prob_inv_rgb)
    H_inv_inf = compute_entropy(prob_inv_inf)
    H_var_rgb = compute_entropy(prob_var_rgb)
    H_var_inf = compute_entropy(prob_var_inf)

    # 损失：最小化 invariant 熵，最大化 variant 熵
    loss_inv = (H_inv_rgb + H_inv_inf).mean()  # 越小越好
    loss_var = - (H_var_rgb + H_var_inf).mean()  # 越大越好 → 加负号变最小化

    return loss_inv + loss_var

#===========================================================上面是特征解耦器的构建==================================================================================#

#===========================================================上面是红外信息注入模块的构建的构建==================================================================================#
class CovarianceCrossAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CovarianceCrossAttention, self).__init__()
        self.channel = in_channels
        self.r = reduction

        # 压缩一下通道，可减少缓存
        self.qkv_reduce = nn.Conv2d(in_channels, in_channels//reduction,1, bias=False)
        self.bn_vis = nn.BatchNorm2d(in_channels//reduction)
        self.bn_ir = nn.BatchNorm2d(in_channels//reduction)

        #恢复通道
        self.restore_vis = nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        self.restore_ir = nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_aug):
        B, C, H, W = x.shape
        R = self.r
        reduced_ch = C//R

        #1. 通道压缩+白化
        vis = self.bn_vis(self.qkv_reduce(x))
        ir = self.bn_ir(self.qkv_reduce(x_aug))

        vis_flat = vis.view(B, reduced_ch, -1)
        ir_flat = ir.view(B, reduced_ch, -1)
        vis_flat = vis_flat-vis.flat.mean(dim=2, keepdim=True)
        ir_feat = ir_flat-ir_flat.mean(dim=2, keepdim=True)

        #2. 互协方差
        cov = torch.bmm(vis_flat, ir_flat.transpose(1, 2))/(H*W-1)

        #3. 一致图
        M= self.sigmoid(f.softmax(cov, dim=-1))

        #4. 把M作用到原始特征上
        vis_att = self.restore_vis(vis)
        ir_att = self.restore_ir(ir)
        m_score = M.mean(dim=[1,2], keepdim=True).view(B,1,1,1)
        F_vis = x * m_score
        F_ir = x_aug *(1-m_score)

        out = F_vis + F_ir + x
        return out




def compute_feature_consistency_loss(x, x_aug):
    triple_loss, entropy_loss, F_inv_rgb, F_inv_inf = FDM(x, x_aug)
    return triple_loss, entropy_loss, F_inv_rgb, F_inv_inf

#================================================域不变信息融合增强=======================================================================#

class SimpleConsistencyFusion(nn.Module):
    """
    不对 F2 做任何卷积处理，直接基于一致性注意力融合 F1 和 F2。
    """

    def __init__(self,inchannels, outchannels):
        super().__init__()
        # 注意：这里没有任何可学习参数（纯注意力融合）

    def forward(self, F_vis, F_inf):
        """
        Args:
            F1: [B, C, H, W]
            F2: [B, C, H, W]
        Returns:
            fused: [B, C, H, W]
        """
        # Step 1: 计算空间一致性注意力图 A ∈ [B, 1, H, W]
        # 使用余弦相似度（也可以换成 L2 距离、点积等）
        sim = f.cosine_similarity(F_vis, F_inf, dim=1).unsqueeze(1)  # [B, 1, H, W]
        A = torch.sigmoid(sim)  # 归一化到 (0, 1)

        # Step 2: 直接加权融合（无任何 F2 变换）
        fused = A * F_vis + (1 - A) * F_inf

        # Step 3（可选）：加残差（比如再加一次 F1）
        out = fused + F_vis
        # 但通常 fused 已经包含 F1，所以一般不需要

        return out

#==========================================================上面是特征融合===============================================================#
class SpaceToDepth(nn.Module):
    def __init__(self, inchannels, outchannels, block_size=2):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size

        # 计算 SpaceToDepth 之后的通道倍率 (k^2)
        # 例如 block_size=2, factor=4
        factor = block_size ** 2

        # 修正：卷积层的输入通道应该是 原始通道 * 倍率
        self.conv = nn.Conv2d(inchannels * factor, outchannels, 1, bias=False)

    def forward(self, x):
        N, C, H, W = x.size()

        # SpaceToDepth 操作
        x = x.view(N, C, H // self.block_size, self.block_size, W // self.block_size, self.block_size)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * (self.block_size ** 2), H // self.block_size, W // self.block_size)

        # 此时 x 的通道数是 C * 4，与 self.conv 的定义 (inchannels * 4) 匹配
        x = self.conv(x)
        return x

class CAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力: [B, C, 1, 1]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        channel_attn = self.sigmoid(out)   # [B, C, 1, 1]

        # 扩展到空间维度: [B, C, H, W]
        channel_attn = channel_attn.expand(-1, -1, x.size(2), x.size(3))

        # 压缩成单通道: [B, 1, H, W]
        attn_map = torch.mean(channel_attn, dim=1, keepdim=True)

        return attn_map


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        out = self.conv(x_cat)  # [B, 1, H, W]
        return out


class SPD_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16, spatial_kernel=3):
        super(SPD_CBAM_Block, self).__init__()

        # Step 1: Space-to-Depth Convolution
        self.spd_conv = SpaceToDepth(block_size=2)

        # Step 2: Channel Attention Module for the first half of channels
        self.cam = CAM(channels=in_channels * 2, reduction_ratio=reduction_ratio)

        # Step 3: Spatial Attention Module for the second half of channels
        self.sam = SAM()

        # Step 4: Concatenation and 1x1 convolution
        self.final_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Step 1: Apply Space-to-Depth
        x_spd = self.spd_conv(x)  # Output shape: [B, 4C, H/2, W/2]

        # Step 2: Split into two groups of 2C each
        split_point = x_spd.shape[1] // 2
        cam_input = x_spd[:, :split_point, :, :]
        sam_input = x_spd[:, split_point:, :, :]

        # Step 3: Apply CAM and SAM on respective halves
        cam_output = self.cam(cam_input)
        sam_output = self.sam(sam_input)

        # Step 4: Concatenate the outputs
        concat_output = torch.cat([cam_output, sam_output], dim=1)  # Shape: [B, 4C, H/2, W/2]

        # Step 5: Final 1x1 convolution to reduce channels back to 2C
        final_output = self.final_conv(concat_output)  # Shape: [B, 2C, H/2, W/2]

        return final_output


def visualize_feature_map(feature_map, title="Feature Map", single_channel_idx=None):
    """
    可视化特征图。

    Args:
        feature_map: 特征图张量，形状为 (B, C, H, W)
        title: 图像标题
        single_channel_idx: 如果指定，则仅显示该通道；否则显示所有通道的网格图
    """
    # 检查维度
    if len(feature_map.shape) != 4:
        raise ValueError("Expected input tensor of shape (B, C, H, W)")

    # 将特征图从 Tensor 转换为 NumPy 数组，并调整范围到 [0, 1]
    feature_map_np = feature_map.detach().cpu().numpy()
    feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min() + 1e-8)

    B, C, H, W = feature_map_np.shape

    if single_channel_idx is not None:
        # 只可视化一个通道
        if single_channel_idx >= C or single_channel_idx < 0:
            raise ValueError(f"Channel index {single_channel_idx} out of bounds")

        fig, axes = plt.subplots(1, B, figsize=(5 * B, 5))
        for b in range(B):
            ax = axes if B == 1 else axes[b]
            ax.imshow(feature_map_np[b, single_channel_idx], cmap='viridis')
            ax.set_title(f"{title} - Batch {b}, Channel {single_channel_idx}")
            ax.axis('off')
    else:
        # 可视化所有通道的网格图
        # 假设我们只展示第一个样本的所有通道
        grid_img = make_grid(torch.tensor(feature_map_np[0]), nrow=int(np.sqrt(C)), normalize=True, scale_each=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis('off')

    plt.show()

class SPDConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(SPDConv,self).__init__()
        self.spd_conv = SpaceToDepth(in_channels,out_channels, block_size=2)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self,x):
        x1 = self.spd_conv(x)
        x2 = self.conv(x)
        output = x1 + x2
        return output
class GA_Concat(nn.Module):
    def __init__(self, channels):
        super(GA_Concat, self).__init__()
        self.cam = CAM(channels=channels)
        self.sam = SAM()

    def forward(self, x):
        deep_feature, shallow_feature = x[0], x[1]

        # 保证 deep_feature 通道数可以被 2 整除
        assert deep_feature.shape[1] % 2 == 0, \
            f"deep_feature channels must be even, but got {deep_feature.shape[1]}"

        cam_x = self.cam(shallow_feature)
        sam_x = self.sam(shallow_feature)

        feat1, feat2 = torch.chunk(deep_feature, 2, dim=1)
        output1 = cam_x * feat1
        output2 = sam_x * feat2
        output = torch.cat([output1, output2], dim=1)

        return output



