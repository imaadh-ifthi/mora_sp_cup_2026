"""Pair loading + crops + geometric augmentation + optional synthetic re-noising."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_image


def list_pairs(gt_dir, noisy_dir):
    gt_dir, noisy_dir = Path(gt_dir), Path(noisy_dir)
    pairs = []
    for gt in sorted(gt_dir.glob("*.png")):
        stem = gt.stem
        for cand in (f"{stem}_noise.png", f"{stem} noise.png",
                     f"{stem}-noise.png", f"{stem}_noisy.png"):
            p = noisy_dir / cand
            if p.exists():
                pairs.append((str(gt), str(p)))
                break
    return pairs


def _geom(a, k, fh, fv):
    if k:
        a = np.ascontiguousarray(np.rot90(a, k))
    if fh:
        a = np.ascontiguousarray(a[:, ::-1])
    if fv:
        a = np.ascontiguousarray(a[::-1, :])
    return a


class PairDataset(Dataset):
    def __init__(self, pairs, crop=224, augment=True, synth=None, synth_prob=0.5, seed=0):
        self.pairs, self.crop, self.augment = pairs, crop, augment
        self.synth, self.synth_prob = synth, synth_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        gt_path, noisy_path = self.pairs[i]
        gt = read_image(gt_path)
        use_synth = self.synth is not None and self.rng.random() < self.synth_prob
        noisy = None if use_synth else read_image(noisy_path)
        H, W, _ = gt.shape
        c = self.crop
        if c and H >= c and W >= c and (H > c or W > c):
            top = int(self.rng.integers(0, H - c + 1))
            left = int(self.rng.integers(0, W - c + 1))
            gt = gt[top:top + c, left:left + c]
            if noisy is not None:
                noisy = noisy[top:top + c, left:left + c]
        if self.augment:
            k = int(self.rng.integers(0, 4))
            fh, fv = self.rng.random() < 0.5, self.rng.random() < 0.5
            gt = _geom(gt, k, fh, fv)
            if noisy is not None:
                noisy = _geom(noisy, k, fh, fv)
        if noisy is None:
            noisy = self.synth(gt, self.rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a).transpose(2, 0, 1)).float()
        return t(noisy), t(gt)
