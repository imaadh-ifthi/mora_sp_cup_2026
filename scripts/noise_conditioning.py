"""Noise conditioning & signal processing noise map estimation.
Implements offline wavelet-MAD noise scale estimation and local sigma map generation.
"""
import numpy as np
import torch
import cv2


def estimate_sigma_mad(img):
    """Estimates noise standard deviation (sigma) using Wavelet Median Absolute Deviation (MAD).
    Input: img as float32 RGB or Grayscale [0, 1]. Returns float sigma.
    """
    if img.ndim == 3:
        # Convert to luminance
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.copy()

    # High-frequency subband approximation using diagonal differences
    # H = [[1, -1], [-1, 1]] / 2
    diff = gray[:-1, :-1] - gray[1:, :-1] - gray[:-1, 1:] + gray[1:, 1:]
    mad = np.median(np.abs(diff))
    sigma = float(mad / (0.6745 * 2.0))
    return max(sigma, 1e-4)


def compute_local_sigma_map(img, win_size=15):
    """Computes a local noise sigma map using sliding-window local variance.
    Input: img as float32 RGB [H, W, 3] or Grayscale [H, W].
    Output: sigma map [H, W, 1] float32.
    """
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)
    mean = cv2.blur(gray, (win_size, win_size))
    mean_sq = cv2.blur(gray * gray, (win_size, win_size))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    sigma = np.sqrt(var)
    return sigma[:, :, None].astype(np.float32)
