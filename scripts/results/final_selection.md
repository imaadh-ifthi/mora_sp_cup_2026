# Final Model & Pipeline Selection Report

## Overview
Evaluated candidate models and TTA options on the 29-pair validation set to select the optimal submission pipeline.

## Evaluation Results

| Configuration | Val dPSNR | Val dSSIM | Val Composite | CPU sec/image | Selected |
|---|---|---|---|---|---|
| `nafnet_medium_raw_dihedral8` | `+8.93 dB` | `+0.402` | `0.5164` | `3.76s` | **YES** |
| `nafnet_medium_ema_dihedral8` | `+8.91 dB` | `+0.401` | `0.5150` | `3.76s` | **NO** |
| `nafnet_medium_raw_flip4` | `+8.90 dB` | `+0.401` | `0.5148` | `2.02s` | **NO** |
| `nafnet_medium_ema_flip4` | `+8.88 dB` | `+0.399` | `0.5134` | `2.09s` | **NO** |
| `insurance_best_dihedral8` | `+8.86 dB` | `+0.392` | `0.5108` | `2.91s` | **NO** |
| `insurance_best_flip4` | `+8.81 dB` | `+0.389` | `0.5080` | `1.80s` | **NO** |
| `nafnet_medium_raw_none` | `+8.71 dB` | `+0.392` | `0.5045` | `1.05s` | **NO** |
| `nafnet_medium_ema_none` | `+8.70 dB` | `+0.390` | `0.5032` | `1.00s` | **NO** |
| `insurance_best_none` | `+8.51 dB` | `+0.374` | `0.4898` | `0.88s` | **NO** |

## Selection Rationale
- **Winning Configuration:** `nafnet_medium_raw_dihedral8`
- **Model Weights Path:** `scripts\weights\nafnet_medium_v2_raw_best.pth`
- **Validation Composite Score:** `0.5164` ($\Delta	ext{PSNR} = +8.93	ext{ dB}$, $\Delta	ext{SSIM} = +0.402$)
- **Score Improvement over Insurance Model:** `+0.0266`
- **GPU Runtime:** `3.76s / image`
