"""Advanced loss functions for low-light image denoising.
Includes: Charbonnier Loss, SSIM Loss, Sobel Gradient Loss, and Log FFT Magnitude Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import SSIMLoss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class SobelGradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x, y):
        # Apply Sobel filters to each channel
        C = x.shape[1]
        loss = 0.0
        for c in range(C):
            xc, yc = x[:, c:c+1], y[:, c:c+1]
            gx_x = F.conv2d(xc, self.sobel_x, padding=1)
            gy_x = F.conv2d(xc, self.sobel_y, padding=1)
            gx_y = F.conv2d(yc, self.sobel_x, padding=1)
            gy_y = F.conv2d(yc, self.sobel_y, padding=1)
            loss += (gx_x - gx_y).abs().mean() + (gy_x - gy_y).abs().mean()
        return loss / C


class LogFFTLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, norm="backward")
        y_fft = torch.fft.rfft2(y, norm="backward")
        x_mag = torch.log(torch.abs(x_fft) + self.eps)
        y_mag = torch.log(torch.abs(y_fft) + self.eps)
        return F.l1_loss(x_mag, y_mag)


class CompositeLossV2(nn.Module):
    def __init__(self, charbonnier_weight=1.0, ssim_weight=0.2, grad_weight=0.05, fft_weight=0.01):
        super().__init__()
        self.char_w = charbonnier_weight
        self.ssim_w = ssim_weight
        self.grad_w = grad_weight
        self.fft_w = fft_weight

        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = SobelGradLoss()
        self.fft_loss = LogFFTLoss()

    def forward(self, pred, gt):
        l_char = self.char_loss(pred, gt) if self.char_w > 0 else 0.0
        l_ssim = self.ssim_loss(pred, gt) if self.ssim_w > 0 else 0.0
        l_grad = self.grad_loss(pred, gt) if self.grad_w > 0 else 0.0
        l_fft = self.fft_loss(pred, gt) if self.fft_w > 0 else 0.0
        return self.char_w * l_char + self.ssim_w * l_ssim + self.grad_w * l_grad + self.fft_w * l_fft
