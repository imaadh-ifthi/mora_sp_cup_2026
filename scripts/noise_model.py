"""Calibrated synthetic low-light sensor noise (classical model) for augmentation.
sRGB -> linearize(^g) -> dark exposure e -> WB jitter -> Poisson shot + Gaussian read
-> ISP gain (1/e, amplifies noise) -> gamma encode -> quantize.
Severity s in [0,1] jointly drives exposure, photon budget and read noise."""
import numpy as np


def _lerp(a, b, t):
    return a + (b - a) * t


class SyntheticNoiseModel:
    def __init__(self, gamma=(2.0, 2.4), exposure=(0.12, 1.0), photon_k=(200.0, 8000.0),
                 read=(0.0005, 0.012), wb_jitter=0.06, quantize=True):
        self.gamma, self.exposure, self.photon_k = gamma, exposure, photon_k
        self.read, self.wb_jitter, self.quantize = read, wb_jitter, quantize

    def __call__(self, clean, rng=None, severity=None):
        rng = rng or np.random.default_rng()
        s = rng.uniform(0.0, 1.0) if severity is None else float(severity)
        g = rng.uniform(*self.gamma)
        e = float(np.exp(_lerp(np.log(self.exposure[1]), np.log(self.exposure[0]), s)))
        K = float(np.exp(_lerp(np.log(self.photon_k[1]), np.log(self.photon_k[0]), s)))
        read = _lerp(self.read[1], self.read[0], s)
        wb = rng.uniform(1.0 - self.wb_jitter, 1.0 + self.wb_jitter, size=(1, 1, 3))
        wb = wb / wb[..., 1:2]
        lin = np.clip(clean, 0.0, 1.0) ** g
        signal = np.clip(lin * e * wb, 0.0, None)
        noisy = rng.poisson(np.maximum(signal * K, 0.0)) / K
        noisy = noisy + rng.normal(0.0, read, size=signal.shape)
        noisy = np.clip(noisy, 0.0, None) / e
        out = np.clip(noisy, 0.0, None) ** (1.0 / g)
        out = np.clip(out, 0.0, 1.0)
        if self.quantize:
            out = np.round(out * 255.0) / 255.0
        return out.astype(np.float32)
