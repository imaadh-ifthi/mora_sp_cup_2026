"""Official-formula metrics + differentiable SSIM loss."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a, b):
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))


def composite(gt, noisy, den):
    p_n, p_d = psnr(noisy, gt), psnr(den, gt)
    s_n, s_d = ssim(noisy, gt), ssim(den, gt)
    dp, ds = p_d - p_n, s_d - s_n
    N = float(np.clip(dp / 15.0, 0.0, 1.0))
    S = float(max(ds, 0.0))
    return {"psnr_noisy": p_n, "psnr_den": p_d, "ssim_noisy": s_n, "ssim_den": s_d,
            "dPSNR": dp, "dSSIM": ds, "N": N, "S": S, "score": 0.6 * N + 0.4 * S}


class SSIMLoss(nn.Module):
    def __init__(self, win_size=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(win_size, dtype=torch.float32) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, win_size, win_size)
        self.register_buffer("kernel", kernel)
        self.register_buffer("rgb", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.pad = win_size // 2

    def _ssim(self, x, y):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        k, p = self.kernel, self.pad
        mu_x, mu_y = F.conv2d(x, k, padding=p), F.conv2d(y, k, padding=p)
        sx = F.conv2d(x * x, k, padding=p) - mu_x * mu_x
        sy = F.conv2d(y * y, k, padding=p) - mu_y * mu_y
        sxy = F.conv2d(x * y, k, padding=p) - mu_x * mu_y
        return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))

    def forward(self, x, y):
        xg = (x * self.rgb).sum(1, keepdim=True)
        yg = (y * self.rgb).sum(1, keepdim=True)
        return 1.0 - self._ssim(xg, yg).mean()
