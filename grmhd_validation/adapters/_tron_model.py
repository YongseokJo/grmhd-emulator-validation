"""Architecture port by the validation implementer.
Source: Kevin Fortune (TRON originals: Sachin), Fall26/tron/model.py,
fall26-tron-rc300 @ 2667a5f. Selected architecture definitions only; construction, checkpoint
loading and physical decode belong to the benchmark adapter. No data access.
"""
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        mid_c = (in_c * 3 * 3 * 3) // (3 * 3 + 3)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1,3,3), stride=(1,stride,stride), padding=0)
        self.bn1 = nn.GroupNorm(8, mid_c)
        self.depth = nn.Conv3d(mid_c, out_c, kernel_size=(3,1,1), stride=(stride,1,1), padding=0)
        self.bn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        # 1. Spatial Padding (Theta: open outflow row-copy; Phi: circular)
        x_spat = F.pad(x, (1, 1, 0, 0, 0, 0), mode='circular') # Phi
        x_spat = torch.cat([x_spat[:, :, :, 0:1, :], x_spat, x_spat[:, :, :, -1:, :]], dim=3) # Theta
        
        x_mid = self.act(self.bn1(self.spatial(x_spat)))
        
        # 2. Depth Padding (R)
        x_dep = F.pad(x_mid, (0, 0, 0, 0, 1, 1), mode='replicate')
        return self.act(self.bn2(self.depth(x_dep)))

class ResidualBlock2Plus1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = Conv2Plus1D(c, c)
        self.c2 = Conv2Plus1D(c, c)
    def forward(self, x): return x + self.c2(self.c1(x))

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, modes, dtype=torch.float32))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, modes, dtype=torch.float32))

    def complex_mult3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_ft = torch.fft.rfftn(x.to(torch.float32), dim=[-3,-2,-1])
        out_ft = torch.zeros(b, self.out_channels, d, h, w//2 + 1, dtype=torch.cfloat, device=x.device)
        
        m1, m2, m3 = min(self.modes, d), min(self.modes, h), min(self.modes, w//2 + 1)
        weights = torch.complex(self.weights_real, self.weights_imag)
        w_slice = weights[:, :, :m1, :m2, :m3]
        out_ft[:, :, :m1, :m2, :m3] = self.complex_mult3d(x_ft[:, :, :m1, :m2, :m3], w_slice)
        
        return torch.fft.irfftn(out_ft, s=(d, h, w)).to(dtype=x.dtype)

class TRON(nn.Module):
    """TRON: Topology-aware Relativistic Operator Network for 3D GRMHD forecasting."""
    def __init__(self, in_c):
        super().__init__()
        self.inc = nn.Sequential(nn.Conv3d(in_c+3, 32, 1), ResidualBlock2Plus1D(32))
        self.down1 = nn.Sequential(Conv2Plus1D(32, 64, stride=2), ResidualBlock2Plus1D(64))
        self.down2 = nn.Sequential(Conv2Plus1D(64, 128, stride=2), ResidualBlock2Plus1D(128))
        self.down3 = nn.Sequential(Conv2Plus1D(128, 256, stride=2), ResidualBlock2Plus1D(256))
        
        self.fno_bot = SpectralConv3d(256, 256, modes=8)
        self.res_bot = ResidualBlock2Plus1D(256)
        
        self.up1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv1 = ResidualBlock2Plus1D(128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv2 = ResidualBlock2Plus1D(64)
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv3 = ResidualBlock2Plus1D(32)
        self.outc = nn.Conv3d(32, in_c, 1)

        # Near-zero init (std=1e-3): breaks identity symmetry on step 1 without exploding
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.outc.bias)

    def _get_grid(self, b, d, h, w, device, dtype):
        z, y, x = torch.meshgrid(torch.arange(d, device=device),
                                 torch.arange(h, device=device),
                                 torch.arange(w, device=device), indexing='ij')
        grid = torch.stack([x/max(w-1, 1), y/max(h-1, 1), z/max(d-1, 1)], dim=0).unsqueeze(0)
        return grid.to(dtype).expand(b, -1, -1, -1, -1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        coords = self._get_grid(b, d, h, w, x.device, x.dtype)
        x_in = torch.cat([x, coords], dim=1)
        
        pd, ph, pw = (8 - d % 8) % 8, (8 - h % 8) % 8, (8 - w % 8) % 8
        x_pad = F.pad(x_in, (0, pw, 0, ph, 0, pd)) 
       
        x1 = checkpoint(self.inc, x_pad, use_reentrant=False)
        x2 = checkpoint(self.down1, x1, use_reentrant=False)
        x3 = checkpoint(self.down2, x2, use_reentrant=False)
        x4 = checkpoint(self.down3, x3, use_reentrant=False)
        
        x4_fno = checkpoint(self.fno_bot, x4, use_reentrant=False)
        x4_res = checkpoint(self.res_bot, x4_fno, use_reentrant=False)
        
        u = checkpoint(self.up1, x4_res, use_reentrant=False)
        u = checkpoint(self.conv1, u + x3, use_reentrant=False)
        u = checkpoint(self.up2, u, use_reentrant=False)
        u = checkpoint(self.conv2, u + x2, use_reentrant=False)
        u = checkpoint(self.up3, u, use_reentrant=False)
        u = checkpoint(self.conv3, u + x1, use_reentrant=False)
        
        out = self.outc(u)
        return out[:, :, :d, :h, :w]
