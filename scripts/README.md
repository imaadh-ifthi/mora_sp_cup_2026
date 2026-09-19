# fit — Low-Light Denoising (MoraSPCup 2026)

Hybrid pipeline combining a NAFNet-lite neural backbone, dihedral-8 deterministic Test-Time Augmentation (TTA), synthetic Poisson-Gaussian noise modeling, offline wavelet-MAD noise map estimation, and a guaranteed classical CPU fallback.

## Quickstart / Entry Point
Run from repository root:

```bash
C:\Users\User\anaconda3\envs\image_proc_lab\python.exe scripts\denoise.py --noise_dir competition_data\submissions\noisy --denoised_dir competition_data\submissions\denoised
```

Inputs `461_noise.png` – `480_noise.png` are denoised and written as `461.png` – `480.png`. The pipeline is 100% offline. If PyTorch or model weights are missing, it automatically triggers the classical CPU fallback (wavelet-MAD + NLM + wavelet shrinkage).

## Manifest & Model Weights (Frozen Artifacts)

| Item | Value |
|---|---|
| Manifest File | `scripts/weights/manifest.json` |
| Primary Weights File | `scripts/weights/best.pth` |
| Insurance Weights File | `scripts/weights/insurance_best.pth` |
| Primary SHA-256 | `88420885d3353e707b73a052919ae5f64dc42ea0ba8f91e76897f705e4204fef` |
| Insurance SHA-256 | `8743c446aab298036761183b92dfcd27a50d313391a10d9756f205079a48f5d0` |
| Google Drive Upload Link | `<GOOGLE_DRIVE_LINK_PLACEHOLDER>` |

## Performance & Benchmark Summary

- **Baseline Score to Beat:** `0.2600`
- **Insurance Checkpoint Score:** `0.4502`
- **Trained Model Validation Score (100 epochs):** `0.4692`
- **Final Validation Score with Dihedral-8 TTA:** **`0.4714`** ($\Delta\text{PSNR} = +8.45\text{ dB}$, $\Delta\text{SSIM} = +0.334$)
- **GPU Inference Speed (RTX 3050):** `2.4s / image` with Dihedral-8 TTA (`< 0.5s` without TTA)
- **CPU Deep Inference Speed:** `1.8s / image`
- **CPU Classical Fallback Speed:** `0.7s / image` (Total for 20 images: ~14s)

## Frozen Version Details
- **Git Commit SHA:** `53bfbbaeae84daf6905f79c4f0e41dad5ab2a413`
- **Report Document:** `fit_Report.pdf` (generated from `scripts/results/report_notes.md`)
