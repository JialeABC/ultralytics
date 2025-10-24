import torch
import torch.nn as nn
import numpy as np
from ultralytics.nn.extra_modules.convolution import Conv,add_conv
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class SobelConv(nn.Module):
    def __init__(self, channel,outchannel) -> None:
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
        l, m, s = x[0], x[1], x[2]
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

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
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


