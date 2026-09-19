"""Deterministic Test-Time Augmentation (TTA) module for image denoising.
Supports: none, flip2, flip4, dihedral8.
"""
import torch
import numpy as np


def apply_transform(img, mode_idx):
    """Applies transformation index 0..7 to a numpy RGB float image [H, W, 3] or tensor [C, H, W]."""
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        # img shape [C, H, W]
        rot_k = mode_idx % 4
        flip_h = (mode_idx >= 4)
        out = torch.rot90(img, rot_k, [1, 2])
        if flip_h:
            out = torch.flip(out, [2])
        return out
    else:
        # numpy shape [H, W, C]
        rot_k = mode_idx % 4
        flip_h = (mode_idx >= 4)
        out = np.rot90(img, rot_k, (0, 1))
        if flip_h:
            out = np.fliplr(out)
        return np.ascontiguousarray(out)


def invert_transform(img, mode_idx):
    """Inverts transformation index 0..7 for numpy RGB float image [H, W, 3] or tensor [C, H, W]."""
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        rot_k = mode_idx % 4
        flip_h = (mode_idx >= 4)
        out = img
        if flip_h:
            out = torch.flip(out, [2])
        out = torch.rot90(out, -rot_k, [1, 2])
        return out
    else:
        rot_k = mode_idx % 4
        flip_h = (mode_idx >= 4)
        out = img
        if flip_h:
            out = np.fliplr(out)
        out = np.rot90(out, -rot_k, (0, 1))
        return np.ascontiguousarray(out)


def get_tta_indices(tta_mode):
    mode = (tta_mode or "none").lower()
    if mode in ("none", "off"):
        return [0]
    elif mode == "flip2":
        return [0, 4]  # identity, horizontal flip
    elif mode == "flip4":
        return [0, 2, 4, 6]  # identity, rot180, hflip, hflip+rot180 (vflip)
    elif mode == "dihedral8":
        return list(range(8))
    else:
        raise ValueError(f"Unknown TTA mode: {tta_mode}")


def tta_tiled_inference(predict_fn, noisy_img, tile=384, stride=320, tta_mode="none"):
    """Runs tiled inference across all TTA variants and averages the reconstructed outputs."""
    from utils import tiled_inference
    indices = get_tta_indices(tta_mode)
    if len(indices) == 1:
        return tiled_inference(predict_fn, noisy_img, tile=tile, stride=stride)

    acc = np.zeros_like(noisy_img, dtype=np.float32)
    for idx in indices:
        t_noisy = apply_transform(noisy_img, idx)
        t_den = tiled_inference(predict_fn, t_noisy, tile=tile, stride=stride)
        den = invert_transform(t_den, idx)
        acc += den

    return acc / float(len(indices))
