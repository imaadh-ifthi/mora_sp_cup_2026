# Competition Report Notes — Team "fit" (MoraSPCup 2026)

## 1. Problem Summary
Low-light image denoising requires removing heavy Poisson-Gaussian sensor noise from low-exposure captures without introducing over-smoothing, severe color shifts, or high-frequency boundary artifacts. The competition benchmark measures performance via a composite score:
$$\text{Composite Score} = 0.6 \times N + 0.4 \times S$$
where $N = \text{clip}(\Delta\text{PSNR} / 15.0, 0, 1)$ and $S = \max(\Delta\text{SSIM}, 0)$. The objective is to maximize restoration quality while meeting strict submission constraints and maintaining offline execution compatibility.

## 2. Dataset Observations & Empirical Analysis
- **Dataset Size:** 460 public high-resolution training pairs (`ground_truth` and `noisy`).
- **Brightness Conservation:** Mean brightness ratio ($\text{noisy} / \text{GT}$) is $0.9984 \approx 1.0$, indicating zero-mean additive sensor noise without systemic underexposure darkening.
- **Noise Severity Range:** Estimated wavelet-MAD standard deviation ($\hat{\sigma}$) spans from $0.0436$ (mild noise) to $0.1424$ (severe grain), with an average $\hat{\sigma} = 0.0892$.
- **Residual Distribution:** Residual error ($\text{noisy} - \text{GT}$) displays signal-dependent heteroscedasticity, fitting a Poisson-Gaussian noise model $y \sim \mathcal{N}(x, \sigma_p^2 x + \sigma_r^2)$.

## 3. Architecture & Neural Design
- **Backbone Network:** **NAFNet-medium** (`widths=(24, 48, 96, 192)`, `blocks=(1, 1, 1, 1)`). Expanded channel capacity by 2.25x relative to `nafnet_small`, capturing fine-grained texture details without memory saturation.
- **Activation-Free Design:** Replaces computationally expensive activation functions (GELU, ReLU) with element-wise multiplication (`SimpleGate`).
- **Channel Attention:** Simplified Channel Attention (SCA) captures global inter-channel relationships without spatial compression.
- **Tiled Sliding Window Inference:** Sliding window tiled inference (`tile=384`, `stride=320`) with linear seam blending prevents memory OOM on RTX 3050 GPUs and eliminates border artifacts.
- **Deterministic Test-Time Augmentation (TTA):** Dihedral-8 group symmetry transformations (identity, 90°/180°/270° rotations, and horizontal flips) applied during inference.

## 4. Classical Signal-Processing Components
- **Wavelet-MAD Noise Estimator:** Computes noise scale $\hat{\sigma}$ using high-frequency diagonal wavelet subband differences ($H = \frac{1}{2}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}$) and Median Absolute Deviation (MAD):
  $$\hat{\sigma} = \frac{\text{median}(|\text{subband}|)}{0.6745 \times 2.0}$$
- **Severity-Balanced Weighted Sampling:** Training samples are dynamically weighted using normalized wavelet-MAD noise estimates ($\text{weight} = 1.0 + 0.7 \times \hat{\sigma}_{\text{norm}}$), ensuring equal model capacity allocation across mild, moderate, and severe noise levels.
- **Classical CPU Fallback Pipeline:** Standalone non-local means (NLM) filtering combined with soft-thresholded stationary wavelet shrinkage ($1.1\text{s/image}$) when PyTorch/GPU is unavailable.
- **Sobel & Log FFT Loss Terms:** Structural gradient and frequency-domain L1 penalties in `CompositeLossV2`.

## 5. Training Methodology & Regularization
- **Loss Function:** Multi-term `CompositeLossV2`:
  $$\mathcal{L} = 1.0 \times \mathcal{L}_{\text{Charbonnier}} + 0.2 \times \mathcal{L}_{\text{SSIM}} + 0.03 \times \mathcal{L}_{\text{Sobel}} + 0.005 \times \mathcal{L}_{\text{LogFFT}}$$
- **Optimization:** AdamW optimizer ($W_{\text{decay}} = 1\times 10^{-4}$, initial $\text{LR} = 2\times 10^{-4}$) with 5-epoch linear warmup + Cosine Annealing Learning Rate scheduling across 120 epochs.
- **Gradient Clipping & Accumulation:** Gradient norm clipping at `0.5`, with 2-step gradient accumulation preserving effective batch size = 8 on `256x256` training crops.
- **Exponential Moving Average (EMA):** Parallel shadow model tracking (`ema_decay = 0.999`) for weight stabilization.
- **Validation Split:** Fixed 6.5% held-out validation split (29 pairs, seed=2026).

## 6. Validation Results & Quantitative Ablation

| Stage / Variant | dPSNR (dB) | dSSIM | Composite Score | Notes |
|---|---|---|---|---|
| Official Baseline | +4.00 dB | +0.100 | 0.2600 | Benchmark cutoff |
| Classical CPU Fallback | +4.20 dB | +0.180 | 0.2400 | Fully offline classical |
| Insurance Model (`nafnet_small` + TTA) | +8.86 dB | +0.392 | 0.5108 | Initial baseline |
| `nafnet_medium_v2_ema` (120 ep) | +8.70 dB | +0.390 | 0.5032 | Single model EMA |
| `nafnet_medium_v2_raw` (120 ep) | +8.71 dB | +0.392 | 0.5045 | Single model RAW |
| **`nafnet_medium_v2_raw` + Dihedral-8 TTA** | **+8.93 dB** | **+0.402** | **0.5164** | **Selected Submission** |

## 7. Alternatives Considered & Technical Rationale
- **BM3D / NLM Only:** Classical denoising lacks representation power for complex color noise and oversmoothens subtle textures.
- **Restormer / Transformer Heavy Networks:** Exceeded RTX 3050 VRAM limits during training and introduced slow CPU fallback times (> 30s/img).
- **Perceptual (VGG) Loss:** VGG perceptual loss introduced checkerboard artifacts and lowered PSNR/SSIM composite metrics relative to direct SSIM loss.

## 8. Runtime & Computational Analysis
- **GPU Inference Time (RTX 3050):**
  - Without TTA: `1.05s per image`
  - With Dihedral-8 TTA: `3.76s per image` (Total for 20 submission images: ~75s)
- **CPU Deep Inference Time:** `1.8s per image`
- **CPU Classical Fallback Time:** `1.1s per image` (Total for 20 submission images: ~22s)
- **VRAM Peak Footprint:** `< 3.2 GB` (RTX 3050 6GB safe)

## 9. Failure Cases & Limitations
- **Extreme Noise ($\hat{\sigma} > 0.15$):** Extremely degraded regions show slight residual high-frequency texture blurring.
- **Chroma Noise in Deep Shadows:** Extremely dark regions occasionally exhibit minor color variance.

## 10. Final Submission Checklist
- [x] Insurance submission verified & fit_insurance.zip created (`Phase 0`).
- [x] Metric alignment verified with official evaluator (`Phase 1`).
- [x] Final submission zip `scripts/submission/fit.zip` contains exactly `461.png` - `480.png`.
- [x] `scripts/denoise.py` accepts `--noise_dir` and `--denoised_dir`.
- [x] SHA-256 checksums documented in `scripts/results/checksums.txt`.
- [x] Offline execution & CPU fallback verified.
