# fit — Low-Light Denoising (MoraSPCup 2026)

Hybrid pipeline: calibrated synthetic-noise augmentation + compact NAFNet-lite,
classical wavelet-MAD noise estimation, and a guaranteed classical CPU fallback.

## Setup (conda env already prepared)
    pip install -r scripts/requirements.txt   # only installs anything if missing

## One-command pipeline (analyze -> train -> denoise -> fit.zip), from REPO_ROOT
    python scripts/run_all.py --team fit

## Inference only (official format, from REPO_ROOT)
    python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
Inputs 461_noise.png–480_noise.png are saved as 461.png–480.png (renaming handled
by the script). Fully offline. Falls back to classical NLM+wavelet if weights/torch missing.

## Model weights (frozen artifact)
| Item | Value |
|---|---|
| File | best.pth |
| Download link | <GOOGLE_DRIVE_LINK> |
| Expected local path | scripts/weights/best.pth |
| SHA-256 | N/A (Classical CPU Fallback) |

## Frozen code version
Git commit SHA: 6fe15af99b2dc5d808cf6004f7cb11df148b9f82

## Training (RTX 3050, AMP): python scripts/train.py --config scripts/config.yaml
Validation uses the exact official composite on a ~30-image hold-out (baseline to beat: 0.26).
Expected runtime: ~5–10 s per 992×992 image on CPU; <1 s on RTX 3050.
