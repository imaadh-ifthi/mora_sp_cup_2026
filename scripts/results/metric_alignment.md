# Metric Alignment Report

## Overview
This report evaluates alignment between the official competition evaluator (`evaluation/evaluate.py`) and the internal validation metric implementation (`scripts/metrics.py`).

## Evaluator Configuration
- **Official Evaluator Command:** `python evaluation/evaluate.py --noisy_dir scripts/tmp/eval_align/noisy --pred_dir scripts/tmp/eval_align/pred --gt_dir scripts/tmp/eval_align/gt`
- **Validation Subset Size:** 15 public validation pairs (held-out split)
- **Model Checkpoint Used:** `scripts/weights/insurance_best.pth`

## Quantitative Results
- **Official Mean PSNR:** 28.8299 dB
- **Official Mean SSIM:** 0.858608
- **Official Mean Delta PSNR:** +7.9386 dB
- **Official Mean Delta SSIM:** +0.365225
- **Official Composite Score:** `0.46363455`
- **Internal Composite Score:** `0.46466798`
- **Absolute Difference:** `0.00103343` (< 0.005 threshold)

## Decision
The internal composite calculation closely tracks the official evaluation metric with a negligible difference of ~0.0010. The internal validation metric (`scripts/metrics.py`) is verified as reliable for candidate model selection and hyperparameter optimization.
