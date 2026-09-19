# fit — Low-Light Denoising (MoraSPCup 2026)

Hybrid pipeline combining a `nafnet_medium` neural backbone (2.25x parameter scaling), dihedral-8 deterministic Test-Time Augmentation (TTA), multi-objective `CompositeLossV2` (Charbonnier + SSIM + Sobel Grad + log FFT), wavelet-MAD severity-balanced sampling, and a guaranteed classical CPU fallback.

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
| Primary Weights File | `scripts/weights/best.pth` (`nafnet_medium_v2_raw_best.pth`) |
| Insurance Weights File | `scripts/weights/insurance_best.pth` |
| Primary SHA-256 | `e8dc418f8ccf630bbe120aacccb444a74518ef869d26fbed73c1b41b79f7dc04` |
| Insurance SHA-256 | `88420885d3353e707b73a052919ae5f64dc42ea0ba8f91e76897f705e4204fef` |
| Google Drive Upload Link | `<GOOGLE_DRIVE_LINK_PLACEHOLDER>` |

## Performance & Benchmark Summary

- **Baseline Score to Beat:** `0.2600`
- **Insurance Model (`nafnet_small` + TTA):** `0.5108`
- **Trained Model Standalone (`nafnet_medium_v2` @ 120 ep):** `0.5045` ($\Delta\text{PSNR} = +8.71\text{ dB}$, $\Delta\text{SSIM} = +0.392$)
- **Final Validation Score with Dihedral-8 TTA:** **`0.5164`** ($\Delta\text{PSNR} = +8.93\text{ dB}$, $\Delta\text{SSIM} = +0.402$)
- **GPU Inference Speed (RTX 3050):** `3.76s / image` with Dihedral-8 TTA (`1.05s` without TTA)
- **CPU Deep Inference Speed:** `1.8s / image`
- **CPU Classical Fallback Speed:** `1.1s / image` (Total for 20 images: ~22s)

## Frozen Version Details
- **Git Commit SHA:** `0c819fb0f3d9f00f7b5d920d05b4f0bcc09922f0`
- **Report Document:** `fit_Report.pdf` (generated from `scripts/results/report_notes.md`)
