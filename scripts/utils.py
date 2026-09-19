"""IO helpers, output-name derivation, tiled inference with linear blending."""
import re
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def write_image(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


_NOISE_SUFFIX = re.compile(r"[\s_\-]?nois(e|y)$", re.I)


def derive_output_name(name):
    """461_noise.png / '461 noise.png' -> 461.png (handbook renaming rule)."""
    return _NOISE_SUFFIX.sub("", Path(name).stem) + ".png"


def _ramp(n, ov):
    w = np.ones(n, np.float32)
    ov = max(int(ov), 1)
    ramp = np.linspace(1.0 / ov, 1.0, ov, dtype=np.float32)
    w[:ov] = ramp
    w[-ov:] = ramp[::-1]
    return w


def tiled_inference(predict_fn, img, tile=384, stride=320, batch=4):
    """predict_fn: (B,3,H,W) tensor -> (B,3,H,W) tensor. Overlap-blended tiling."""
    import torch
    H, W, _ = img.shape
    tile = min(tile, H, W)
    stride = min(stride, stride)
    ov = tile - stride

    def pad_len(n):
        return tile - n if n <= tile else (stride - (n - tile) % stride) % stride

    ph, pw = pad_len(H), pad_len(W)
    imgp = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else img
    Hp, Wp = imgp.shape[:2]
    ys = list(range(0, Hp - tile + 1, stride)) or [0]
    xs = list(range(0, Wp - tile + 1, stride)) or [0]
    win = np.outer(_ramp(tile, ov), _ramp(tile, ov))

    acc = np.zeros((3, Hp, Wp), np.float32)
    wacc = np.zeros((Hp, Wp), np.float32)
    tiles, positions = [], []
    for y in ys:
        for x in xs:
            tiles.append(imgp[y:y + tile, x:x + tile])
            positions.append((y, x))

    for i in range(0, len(tiles), batch):
        chunk = np.transpose(np.stack(tiles[i:i + batch]), (0, 3, 1, 2))
        out = predict_fn(torch.from_numpy(chunk)).cpu().numpy().astype(np.float32)
        for j, (y, x) in enumerate(positions[i:i + batch]):
            acc[:, y:y + tile, x:x + tile] += out[j] * win
            wacc[y:y + tile, x:x + tile] += win

    den = acc / np.maximum(wacc, 1e-6)[None]
    return np.clip(den[:, :H, :W].transpose(1, 2, 0), 0, 1)
