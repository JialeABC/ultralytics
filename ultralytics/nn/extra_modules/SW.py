import torch
import torch.nn as nn
from torch.nn import Module

class SwitchWhiten2d(Module):
    def __init__(self, num_features, num_pergroup=16, sw_type=2, T=5, tie_weight=False, eps=1e-5, momentum=0.99, affine=True):
        super(SwitchWhiten2d, self).__init__()
        if sw_type not in [2, 3, 5]:
            raise ValueError('sw_type should be in [2, 3, 5]')
        assert num_features % num_pergroup == 0
        self.num_features = num_features
        self.num_pergroup = num_pergroup
        self.num_groups = num_features // num_pergroup
        self.sw_type = sw_type
        self.T = T
        self.tie_weight = tie_weight
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        num_components = sw_type

        self.sw_mean_weight = nn.Parameter(torch.ones(num_components))
        self.sw_var_weight = nn.Parameter(torch.ones(num_components)) if not tie_weight else None
        self.weight = nn.Parameter(torch.ones(num_features)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if affine else None

        self.register_buffer('running_mean', torch.zeros(self.num_groups, num_pergroup, 1))
        self.register_buffer('running_cov', torch.eye(num_pergroup).unsqueeze(0).repeat(self.num_groups, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        c, g = self.num_pergroup, self.num_groups

        in_data_t = x.transpose(0, 1).contiguous().view(g, c, -1)
        if self.training:
            mean_bn = in_data_t.mean(-1, keepdim=True)
            in_data_bn = in_data_t - mean_bn
            cov_bn = torch.bmm(in_data_bn, in_data_bn.transpose(1, 2)).div(H*W*N)
            self.running_mean.mul_(self.momentum).add_((1-self.momentum)*mean_bn.data)
            self.running_cov.mul_(self.momentum).add_((1-self.momentum)*cov_bn.data)
        else:
            mean_bn = self.running_mean
            cov_bn = self.running_cov

        mean_bn = mean_bn.view(1,g,c,1).expand(N,g,c,1).contiguous().view(N*g,c,1)
        cov_bn = cov_bn.view(1,g,c,c).expand(N,g,c,c).contiguous().view(N*g,c,c)
        in_data = x.view(N*g,c,-1)
        eye = torch.eye(c, device=x.device).view(1,c,c).expand(N*g,c,c)
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        cov_in = torch.bmm(x_in, x_in.transpose(1,2)).div(H*W)

        mean_weight = torch.softmax(self.sw_mean_weight, 0)
        var_weight = torch.softmax(self.sw_var_weight, 0) if self.sw_var_weight is not None else mean_weight

        if self.sw_type == 2:
            mean = mean_weight[0] * mean_bn + mean_weight[1] * mean_in
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + self.eps * eye
        else:
            raise NotImplementedError("Only sw_type=2 is used in your pretrained weight")

        Ng, c, _ = cov.size()
        P = torch.eye(c, device=cov.device).expand(Ng, c, c)
        rTr = (cov * P).sum((1,2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P,3), cov_N)
        wm = P * rTr.sqrt()

        x_hat = torch.bmm(wm, in_data - mean).view(N,C,H,W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1,C,1,1) + self.bias.view(1,C,1,1)
        return x_hat