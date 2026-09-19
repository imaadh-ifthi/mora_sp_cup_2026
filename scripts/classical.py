"""Classical DSP: wavelet-MAD noise estimation, VisuShrink, NLM/BM3D CPU fallback."""
import cv2
import numpy as np

try:
    import pywt
except ImportError:
    pywt = None
try:
    import bm3d as _bm3d
except Exception:
    _bm3d = None


def estimate_sigma(img):
    """Robust noise std (MAD of finest wavelet HH subband), in [0,1] units."""
    if pywt is not None:
        sigs = []
        for c in range(img.shape[2]):
            _, (_, _, hh) = pywt.dwt2(img[..., c], "haar", mode="symmetric")
            sigs.append(float(np.median(np.abs(hh)) / 0.6745))
        return float(np.mean(sigs))
    g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))) / 2.0)


def wavelet_shrink(channel, sigma, wavelet="db4", levels=4):
    if pywt is None or sigma <= 0:
        return channel
    coeffs = pywt.wavedec2(channel, wavelet, mode="symmetric", level=levels)
    out = [coeffs[0]]
    n = max(channel.size, 2)
    for details in coeffs[1:]:
        new = []
        for d in details:
            s = np.median(np.abs(d)) / 0.6745
            T = 0.7 * s * np.sqrt(2.0 * np.log(n))
            new.append(pywt.threshold(d, T, mode="soft"))
        out.append(tuple(new))
    rec = pywt.waverec2(out, wavelet, mode="symmetric")
    return rec[:channel.shape[0], :channel.shape[1]]


def cpu_fallback(img, sigma=None, use_bm3d=True):
    """Deterministic, dependency-light denoiser. ALWAYS produces output."""
    if sigma is None:
        sigma = estimate_sigma(img)
    if use_bm3d and _bm3d is not None:
        return np.clip(_bm3d.bm3d_rgb(img, sigma_psd=float(sigma)), 0, 1).astype(np.float32)
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    h = float(np.clip(2.0 * sigma * 255.0, 3.0, 28.0))
    den = cv2.fastNlMeansDenoisingColored(u8, None, h=h, hColor=0.6 * h,
                                          templateWindowSize=7, searchWindowSize=21
                                          ).astype(np.float32) / 255.0
    if pywt is not None:
        ycrcb = cv2.cvtColor(den, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        y = wavelet_shrink(ycrcb[..., 0], sigma)
        cr = wavelet_shrink(ycrcb[..., 1], 1.3 * sigma)
        cb = wavelet_shrink(ycrcb[..., 2], 1.3 * sigma)
        den = cv2.cvtColor(np.stack([y, cr, cb], -1).astype(np.float32), cv2.COLOR_YCrCb2RGB)
    return np.clip(den, 0, 1).astype(np.float32)
