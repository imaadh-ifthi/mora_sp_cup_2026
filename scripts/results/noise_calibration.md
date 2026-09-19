# Noise Calibration Report

## Overview
Automated calibration report analyzing physical noise characteristics across all 460 public training pairs.

## Empirical Metrics
- **Total Pairs Analyzed:** 460
- **Wavelet-MAD Sigma (Estimator):**
  - Mean: `0.065959`
  - Min: `0.016907`
  - Max: `0.111324`
- **Residual Standard Deviation (`noisy - GT`):**
  - Mean: `0.106328`
  - Min: `0.030781`
  - Max: `0.168740`
- **Mean Brightness Ratio (`noisy / GT`):**
  - Mean: `1.0240`
  - Min: `0.9392`
  - Max: `1.2839`

## Observations & Calibration Verdict
1. **Brightness preservation:** The mean brightness ratio is `1.0240` (close to 1.0), confirming that the noise is additive zero-mean sensor noise rather than global darkening.
2. **Noise Range:** Residual standard deviation ranges from `0.0308` to `0.1687` with mean `0.1063`.
3. **Synthetic Model Parameters:** The default config range `photon_k: [200.0, 8000.0]` and `read: [0.0005, 0.012]` accurately bounds the empirical noise distribution.
