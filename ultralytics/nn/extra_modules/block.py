import torch
import torch.nn as nn
import numpy as np
# from ultralytics.nn.extra_modules.convolution import add_conv
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False



    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class LoGConv(nn.Module):
    def __init__(self, channel, sigma=1.0, kernel_size=5) -> None:
        """
        高斯拉普拉斯(LoG)卷积层实现
        Args:
            channel: 输入通道数
            outchannel: 输出通道数（兼容你的Sobel接口，实际实现中保持和输入通道一致）
            sigma: 高斯核的标准差，默认1.0
            kernel_size: LoG核的尺寸，建议为奇数，默认5
        """
        super().__init__()

        # 1. 生成LoG核
        log_kernel = self._create_log_kernel(kernel_size, sigma)
        # 2. 调整核的形状: [kernel_size, kernel_size] -> [channel, 1, 1, kernel_size, kernel_size]
        #    适配3D卷积的权重形状 [out_channels, in_channels/groups, depth, height, width]
        log_kernel_3d = torch.tensor(log_kernel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, kernel_size,
                                                                                          kernel_size)

        # 3. 定义分组3D卷积（保持和Sobel实现一致的卷积类型）
        padding = kernel_size // 2  # 保持输出尺寸和输入一致
        self.log_conv3d = nn.Conv3d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            padding=padding,
            groups=channel,
            bias=False
        )

        # 4. 赋值卷积核权重并冻结（不参与梯度更新）
        self.log_conv3d.weight.data = log_kernel_3d.clone()
        self.log_conv3d.weight.requires_grad = False  # 修正原代码的小问题：应该给weight设置requires_grad

        # 5. 如果需要调整输出通道数，增加1x1卷积（兼容你的outchannel参数）

        self.channel_adjust = nn.Identity()

    def _create_log_kernel(self, kernel_size, sigma):
        """生成高斯拉普拉斯核"""
        # 生成坐标网格
        ax = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        x, y = np.meshgrid(ax, ax)

        # LoG公式: ∇²G = (x² + y² - 2σ²) / (σ⁴) * exp(-(x² + y²)/(2σ²))
        g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        log = (x ** 2 + y ** 2 - 2 * sigma ** 2) / (sigma ** 4) * g

        # 归一化（可选，使核的总和为0，避免改变图像整体亮度）
        log = log / np.sum(np.abs(log))

        return log

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 [B, C, H, W]
        Returns:
            输出张量，形状 [B, outchannel, H, W]
        """
        # 扩展维度适配3D卷积: [B, C, H, W] -> [B, C, 1, H, W]
        x_3d = x[:, :, None, :, :]

        # 执行LoG卷积
        log_out = self.log_conv3d(x_3d)

        # 调整输出通道数
        log_out = self.channel_adjust(log_out)

        # 压缩维度: [B, outchannel, 1, H, W] -> [B, outchannel, H, W]
        return log_out[:, :, 0]


class SIE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc,inc)
        self.conv1 = nn.Conv2d(in_channels=2*inc,out_channels=inc,kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_out = self.conv2(self.conv1(x_concat)+x)

        return x_out


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)
        self.non_strided_conv = nn.Conv2d(in_channels=ouc,out_channels=ouc,kernel_size=3,stride=1,padding=1)
        self.orin_conv = nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3,stride=2,padding=1)


    def forward(self, x):
        x2 = self.orin_conv(x)
        x1 = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x1 = self.conv(x1)
        x1 = self.non_strided_conv(x1)
        x_out = x1 + x2
        return x_out

class Dual_Grad_SPD(nn.Module):
    def __init__(self,inc, dimension=1):
        super().__init__()

        self.d = dimension
        self.conv = Conv(inc * 4, inc * 2, k=3)
        self.fusion_conv = nn.Conv2d(in_channels=inc * 8,out_channels=inc * 4,kernel_size=1,stride=1,padding=0)
        self.non_strided_conv = nn.Conv2d(in_channels=inc * 2, out_channels=inc * 2, kernel_size=3, stride=1, padding=1)

        self.sobel_conv1 = SobelConv(channel=16)
        self.sobel_conv2 = SobelConv(channel=32)
        self.sobel_conv3 = SobelConv(channel=64)
        self.sobel_conv4 = SobelConv(channel=128)

        self.log_conv1 = LoGConv(channel=16)
        self.log_conv2 = LoGConv(channel=32)
        self.log_conv3 = LoGConv(channel=64)
        self.log_conv4 = LoGConv(channel=128)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        f1 = x[..., ::2, ::2]
        f2 = x[..., 1::2, ::2]
        f3 = x[..., ::2, 1::2]
        f4 = x[..., 1::2, 1::2]

        if C==16:
            x1 = self.conv1(x)

            sobel_f1 = self.sobel_conv1(f1)
            sobel_f2 = self.sobel_conv1(f2)
            sobel_f3 = self.sobel_conv1(f3)
            sobel_f4 = self.sobel_conv1(f4)

            log_f1 = self.log_conv1(f1)
            log_f2 = self.log_conv1(f2)
            log_f3 = self.log_conv1(f3)
            log_f4 = self.log_conv1(f4)

        elif C==32:
            x1 = self.conv2(x)

            sobel_f1 = self.sobel_conv2(f1)
            sobel_f2 = self.sobel_conv2(f2)
            sobel_f3 = self.sobel_conv2(f3)
            sobel_f4 = self.sobel_conv2(f4)

            log_f1 = self.log_conv2(f1)
            log_f2 = self.log_conv2(f2)
            log_f3 = self.log_conv2(f3)
            log_f4 = self.log_conv2(f4)

        elif C==64:
            x1 = self.conv3(x)

            sobel_f1 = self.sobel_conv3(f1)
            sobel_f2 = self.sobel_conv3(f2)
            sobel_f3 = self.sobel_conv3(f3)
            sobel_f4 = self.sobel_conv3(f4)

            log_f1 = self.log_conv3(f1)
            log_f2 = self.log_conv3(f2)
            log_f3 = self.log_conv3(f3)
            log_f4 = self.log_conv3(f4)

        elif C==128:
            x1 = self.conv4(x)

            sobel_f1 = self.sobel_conv4(f1)
            sobel_f2 = self.sobel_conv4(f2)
            sobel_f3 = self.sobel_conv4(f3)
            sobel_f4 = self.sobel_conv4(f4)

            log_f1 = self.log_conv4(f1)
            log_f2 = self.log_conv4(f2)
            log_f3 = self.log_conv4(f3)
            log_f4 = self.log_conv4(f4)

        sobel = torch.cat([sobel_f1,sobel_f2,sobel_f3,sobel_f4],1)
        log = torch.cat([log_f1,log_f2,log_f3,log_f4],1)
        x2 = torch.cat([sobel,log],1)
        x3 = self.fusion_conv(x2)
        x4 = self.conv(x3)
        x_out = x1 + x4
        return x_out

class TSDF(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.conv1_l = nn.Conv2d(in_channels=self.in_dim//2, out_channels=self.in_dim, kernel_size=3, stride=2, padding=1)
        self.conv2_l = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1,padding=1)
        self.conv_m = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1)
        self.conv1_s = nn.Conv2d(in_channels=self.in_dim*2, out_channels=self.in_dim, kernel_size=3, stride=2, padding=1)
        self.conv2_s = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1,padding=1)
        self.conv_lms = nn.Conv2d(in_channels=self.in_dim*3,out_channels=self.out_dim,kernel_size=1,stride=1,padding=0)


    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[2], x[1], x[0]
        tgt_size = m.shape[2:]
        l = self.conv1_l(l)  #there is a bug here
        m = self.conv_m(m)
        s = self.conv1_s(s)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv2_l(l)
        #l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        s = self.conv2_s(s)
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m, s], dim=1)
        lms = self.conv_lms(lms)
        return lms



def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, inchannel,outchannel, midchannel,rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.midchannel = midchannel
        # self.dim = [512, 256, 256]
        self.dim = [self.midchannel*2,self.midchannel,self.midchannel//2]
        # if level==0:
        #     self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
        #     self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
        #     self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        # elif level==1:
        #     self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
        #     self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
        #     self.expand = add_conv(self.inter_dim, 512, 3, 1)


        self.compress_level_0 = add_conv(self.dim[0], self.dim[1], 1, 1)
        self.stride_level_2 = add_conv(self.dim[2], self.dim[1], 3, 2)
        self.expand = add_conv(self.dim[1], self.dim[0], 3, 1)
        # elif level==2:
        #     self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
        #     self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.dim[1], compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.dim[1], compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.dim[1], compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x):

        x_level_0 = x[2]
        x_level_1 = x[1]
        x_level_2 = x[0]

        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
        level_1_resized =x_level_1
        level_2_resized =self.stride_level_2(x_level_2)


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out



class ASFF1(nn.Module):
    def __init__(self, inchannel,outchannel, rfb=False, vis=False):
        super(ASFF1, self).__init__()
        self.inter_dim = inchannel

        self.compress_level_0 = add_conv(self.inter_dim*2, self.inter_dim, 1, 1)
        self.expand = add_conv(self.inter_dim, outchannel, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x):
        x_level_0 = x[2]    # smaller feature map
        x_level_1 = x[1]    # middle feature map
        x_level_2 = x[0]    # bigger feature map


        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
        level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
        level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out



class SSFF(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SSFF, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv1 =  Conv(inchannel*2, inchannel,1)
        self.conv2 =  Conv(inchannel*4, inchannel,1)
        self.conv3d = nn.Conv3d(inchannel,outchannel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(outchannel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)  # (1,1,32,64,64)
        p4_3d = torch.unsqueeze(p4_2, -3)  # (1,1,32,64,64)
        p5_3d = torch.unsqueeze(p5_2, -3) # (1,1,32,64,64)
        combine = torch.cat([p3_3d,p4_3d,p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

class Downsample(nn.Module):
    def __init__(self, inchannel,outchannel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Neck_Conv(nn.Module):
    def __init__(self, inchannel,outchannel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x

class C2f_LMSC(nn.Module):
    def __init__(self, inchannel,outchannel):
        super().__init__()
        self.in_dim = inchannel//2
        self.conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=7, stride=1, padding=3)
        self.conv = nn.Conv2d(in_channels=5*self.in_dim, out_channels=self.in_dim*2, kernel_size=1)

    def forward(self, x):
        x1, x2 = x.split(self.in_dim, 1)
        x2_1 = self.conv1(x2)
        x2_2 = self.conv2(x2)
        x2_3 = self.conv3(x2)
        x2_4 = self.conv4(x2)
        x2_out = torch.cat((x2_1, x2_2, x2_3,x2_4), dim=1)
        x = torch.cat((x1, x2_out), dim=1)
        x = self.conv(x)
        return x

class C3_iRMB(nn.Module):
    def __init__(self, inchannel,outchannel):
        super().__init__()
        self.in_dim = inchannel
        self.out_dim = outchannel
        self.conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim,kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(self.in_dim)
        self.attention = nn.Sequential()
        self.dw_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=0)
        self.SE = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x21 = self.conv2(x2)
        x2 = self.conv3(x21)
        x3 = self.batch_norm(x)
        x3 = self.conv3(x3)
        x3_att = self.attention(x3)
        x3 = x3 * x3_att
        x3_1 = self.SE(self.dw_conv(x3))
        x3 = x3 + x3_1
        x3 = self.conv1(x3)
        x_2 = x3 + x2
        x_out = torch.cat((x1,x_2), dim=1)
        return x_out



