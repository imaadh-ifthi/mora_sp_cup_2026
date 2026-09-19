# MoraSPCup 2026 — Low-Light Image Denoising (Team "fit")

Official repository for Team **fit**'s submission to MoraSPCup 2026 Low-Light Image Denoising. 

This solution combines a high-capacity **`nafnet_medium`** neural backbone (2.25$\times$ parameter expansion over baseline), 8-fold Dihedral group Test-Time Augmentation (TTA), multi-objective composite loss optimization, wavelet-MAD severity-balanced weighted sampling, and an offline classical CPU fallback pipeline.

---

## Model Verification
| Item | Value |
|------|-------|
| **Model File** | `scripts/weights/best.pth` |
| **Location** | Committed directly to GitHub Repository |
| **SHA-256 Checksum** | `e8dc418f8ccf630bbe120aacccb444a74518ef869d26fbed73c1b41b79f7dc04` |
| **Git Commit SHA** | `68fd3a2c7ab907c66cce2cc923b2dff292238385` |

*Note: This checksum verifies the integrity of the frozen model used for preliminary submission.*

---

## 1. Exact Run Command & Execution Instructions

### **Prerequisites & Dependencies**
Dependencies are listed in `scripts/requirements.txt`:
- `torch` ($\ge 2.0.0$)
- `torchvision`
- `opencv-python`
- `scikit-image`
- `pyyaml`
- `numpy`

Install dependencies using:
```bash
python -m pip install -r scripts/requirements.txt
```

---

### **Primary Entry Point (Exact Run Command)**
Denoises all noisy test submission images (`461_noise.png` – `480_noise.png`) and writes outputs (`461.png` – `480.png`) to the designated output folder:

```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
```

---

### **CPU Fallback Confirmation & Timing**
- **Tested on CPU:** Confirmed and verified fully operational on CPU.
- **Deep Model CPU Speed:** `1.8s / image` (Total 20 images: ~36 seconds).
- **Classical Fallback Speed:** `1.1s / image` (Total 20 images: ~22 seconds).
- **CPU Fallback Command:**
  ```bash
  python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir scripts/tmp/cpu_fallback_test --device cpu --classical-only
  ```

---

### **Other Execution Commands**
- **Run Candidate Evaluation Grid:**
  ```bash
  python scripts/eval_grid.py
  ```
- **Train `nafnet_medium` from Scratch:**
  ```bash
  python scripts/train_v2.py --config scripts/config_v2.yaml
  ```

---

## 2. Model Architecture Deep-Dive (`nafnet_medium`)

The neural architecture is based on **NAFNet** (*Non-Linear Activation Free Network*, ECCV 2022), optimized specifically for high-efficiency image restoration.

```
                    ┌──────────────────────────────────────────────┐
                    │               Input Noisy Image              │
                    └──────────────────────┬───────────────────────┘
                                           │  3x3 Conv
                                           ▼
                                 ┌──────────────────┐
                                 │ Intro (24 ch)    │
                                 └─────────┬────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼ Encoder 1 (24 ch)                   │ Skip 1
                        │ Downsample (24 -> 48 ch)            │
                        ▼ Encoder 2 (48 ch)                   │ Skip 2
                        │ Downsample (48 -> 96 ch)            │
                        ▼ Encoder 3 (96 ch)                   │ Skip 3
                        │ Downsample (96 -> 192 ch)           │
                        ▼ Encoder 4 / Bottleneck (192 ch)     │
                        └──────────────────┬──────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼ Decoder 4 (192 -> 96 ch) + Skip 3   │
                        ▼ Decoder 3 (96 -> 48 ch)  + Skip 2   │
                        ▼ Decoder 2 (48 -> 24 ch)  + Skip 1   │
                        ▼ Decoder 1 (24 ch)                   │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                                 ┌──────────────────┐
                                 │ Ending (3 ch)    │
                                 └─────────┬────────┘
                                           │  + Input Residual
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │              Output Denoised Image           │
                    └──────────────────────────────────────────────┘
```

### **Architectural Components**
- **Channel Scaling (`widths: [24, 48, 96, 192]`, ~3.5M parameters):** Expands representation capacity by 2.25$\times$ relative to the 0.9M baseline, enabling rich color noise feature extraction without memory saturation.
- **SimpleGate Activation:** Replaces non-linear activations (GELU/ReLU) with element-wise tensor chunking and multiplication:
  $$\text{SimpleGate}(X) = X_1 \odot X_2 \quad \text{where } [X_1, X_2] = \text{chunk}(X, \text{dim}=1)$$
- **Simplified Channel Attention (SCA):** Captures global spatial information via adaptive average pooling paired with 1x1 convolutions, avoiding spatial compression loss.
- **Overlapping Tiled Window Inference (`tile=384`, `stride=320`):** Evaluates full-resolution images via overlapping tiles with linear boundary blending to prevent memory OOM and edge seam artifacts.
- **Dihedral-8 Test-Time Augmentation (TTA):** Runs inference across all 8 elements of the 2D dihedral transformation group (rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ$ and horizontal flips), averaging outputs for maximum structural stability.

### **Training Loss Function (`CompositeLossV2`)**
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.2 \cdot \mathcal{L}_{\text{SSIM}} + 0.03 \cdot \mathcal{L}_{\text{SobelGrad}} + 0.005 \cdot \mathcal{L}_{\text{LogFFT}}$$

---

## 3. Model Weights Details

All trained model weights are committed directly inside `scripts/weights/`:

| Model Checkpoint | File Path | SHA-256 Checksum | Google Drive Backup Link | Description |
|---|---|---|---|---|
| **Primary Model (`best.pth`)** | `scripts/weights/best.pth` | `e8dc418f8ccf630bbe120aacccb444a74518ef869d26fbed73c1b41b79f7dc04` | `<GOOGLE_DRIVE_LINK_PLACEHOLDER>` | Winning `nafnet_medium` checkpoint |
| **EMA Model** | `scripts/weights/nafnet_medium_v2_ema_best.pth` | `feeaf9ccbb0a43759f6414696e4aaa3480e03e9e602eff834bb4cc6232fa1ac8` | `<GOOGLE_DRIVE_LINK_PLACEHOLDER>` | Exponential Moving Average checkpoint |
| **Insurance Model** | `scripts/weights/insurance_best.pth` | `88420885d3353e707b73a052919ae5f64dc42ea0ba8f91e76897f705e4204fef` | `<GOOGLE_DRIVE_LINK_PLACEHOLDER>` | Initial `nafnet_small` fallback |
| **Manifest File** | `scripts/weights/manifest.json` | `2b85be6edad2de557b0d1d0bdb58abb7d383b18bd177374f9bb9fd40324e27a2` | N/A | Model metadata manifest |

---

## 4. Performance & Benchmark Summary

- **Official Baseline Score:** `0.2600`
- **Insurance Model (`nafnet_small` + TTA):** `0.5108`
- **Trained `nafnet_medium_v2` Standalone (120 ep):** `0.5045` ($\Delta\text{PSNR} = +8.71\text{ dB}$, $\Delta\text{SSIM} = +0.392$)
- **Final Selected Score (`nafnet_medium_v2` + Dihedral-8 TTA):** **`0.5164`** ($\Delta\text{PSNR} = +8.93\text{ dB}$, $\Delta\text{SSIM} = +0.402$)
- **GPU Inference Speed (RTX 3050):** `3.76s / image` with Dihedral-8 TTA (`1.05s` without TTA)
- **CPU Deep Inference Speed:** `1.8s / image`
- **CPU Classical Fallback Speed:** `1.1s / image` (Total for 20 images: ~22s)

---

## 5. Repository Integrity & Checksums
- **Git Commit SHA:** `68fd3a2c7ab907c66cce2cc923b2dff292238385`
- **Submission Archive:** `scripts/submission/fit.zip` (SHA-256: `bf2ed6116f414ce317714a988e1987ce9b2d7abd9ee098d09ce36aac725cc9a0`)
- **Report Document:** `scripts/results/report_notes.md`
