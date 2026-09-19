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

## 1. Submission Specifications & Requirements

- **Exact Run Command:**
  ```bash
  python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
  ```

- **Dependencies:**
  Dependencies are listed in [`scripts/requirements.txt`](file:///c:/Competitions/mora_sp_cup_2026/scripts/requirements.txt):
  - `torch` ($\ge 2.0.0$)
  - `torchvision`
  - `opencv-python`
  - `scikit-image`
  - `pyyaml`
  - `numpy`

  To install dependencies:
  ```bash
  python -m pip install -r scripts/requirements.txt
  ```

- **Git Commit SHA:** `68fd3a2c7ab907c66cce2cc923b2dff292238385`

- **Model Info:**
  - **Filename:** `best.pth`
  - **Expected Local Path:** `scripts/weights/best.pth`
  - **Download Link (Google Drive):** Committed directly to GitHub repo at `scripts/weights/best.pth` (Google Drive external mirror backup: `<GOOGLE_DRIVE_LINK_PLACEHOLDER>`).
  - **SHA-256 Checksum:** `e8dc418f8ccf630bbe120aacccb444a74518ef869d26fbed73c1b41b79f7dc04`

- **CPU Fallback Confirmation:**
  - **Confirmation:** The code was explicitly tested on CPU and verified to produce identical valid outputs within reasonable execution time.
  - **Deep Model CPU Speed:** `1.8s / image` (~36 seconds total for all 20 preliminary test images).
  - **Classical CPU Fallback Speed:** `1.1s / image` (~22 seconds total for all 20 preliminary test images).
  - **CPU Test Command:**
    ```bash
    python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir scripts/tmp/cpu_fallback_test --device cpu --classical-only
    ```

---

## 2. Model Architecture & Method Overview (`nafnet_medium`)

The neural architecture is based on **NAFNet** (*Non-Linear Activation Free Network*, ECCV 2022), optimized specifically for high-efficiency image restoration.

### **Key Components**
- **Channel Scaling (`widths: [24, 48, 96, 192]`, ~3.5M parameters):** Expands feature extraction capacity by 2.25$\times$ over baseline.
- **SimpleGate Activation & SCA:** Replaces non-linear activations with element-wise tensor chunking and simplified channel attention.
- **Overlapping Tiled Inference (`tile=384`, `stride=320`):** Seamless full-resolution inference using boundary linear weight blending.
- **Dihedral-8 TTA:** Evaluates all 8 elements of the 2D dihedral group ($0^\circ, 90^\circ, 180^\circ, 270^\circ$ + flips) and averages predictions for state-of-the-art restoration quality.

---

## 3. Project Directory Structure

```text
mora_sp_cup_2026/
├── README.md                       # Main repository overview & verification
├── fit.zip                         # Primary preliminary submission zip
├── competition_data/
│   ├── public/
│   │   ├── ground_truth/
│   │   └── noisy/
│   └── submissions/
│       ├── noisy/                  # Preliminary noisy images (461_noise.png - 480_noise.png)
│       └── denoised/               # Output denoised images (461.png - 480.png)
└── scripts/
    ├── README.md                   # Detailed technical documentation
    ├── denoise.py                  # Official offline inference CLI entry point
    ├── model.py                    # NAFNet model architecture definition
    ├── config_v2.yaml              # Hyperparameter configuration
    ├── train_v2.py                 # Severity-balanced training script
    ├── losses_v2.py                # Multi-objective composite loss functions
    ├── eval_grid.py                # Candidate evaluation & validation benchmark
    ├── requirements.txt            # Python dependencies
    ├── weights/
    │   ├── best.pth                # Primary selected weights (SHA256: e8dc41...)
    │   ├── nafnet_medium_v2_ema_best.pth
    │   ├── insurance_best.pth
    │   └── manifest.json
    ├── submission/
    │   └── fit.zip                 # Submissions folder zip copy
    └── results/
        ├── final_selection.md       # Validation metric results log
        ├── checksums.txt            # All artifact SHA-256 checksums
        └── report_notes.md          # Content structure for 5-page PDF report
```
