"""Comprehensive candidate evaluation grid.
Evaluates insurance_best, nafnet_medium_v2_raw, and nafnet_medium_v2_ema across TTA modes (none, flip4, dihedral8)
on the standardized 29-pair validation set. Outputs final_selection.md.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from dataset import list_pairs
from metrics import composite
from model import build_model
from tta import tta_tiled_inference
from utils import read_image


def run_eval():
    cfg = yaml.safe_load(open(HERE / "config_v2.yaml"))
    seed = cfg["seed"]
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = list_pairs(ROOT / cfg["data_root"] / "ground_truth", ROOT / cfg["data_root"] / "noisy")
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(pairs) * cfg["val_ratio"]))
    val_pairs = [pairs[i] for i in idx[:n_val]]

    candidates = [
        ("insurance_best", "nafnet_small", HERE / "weights" / "insurance_best.pth"),
        ("nafnet_medium_raw", "nafnet_medium", HERE / "weights" / "nafnet_medium_v2_raw_best.pth"),
        ("nafnet_medium_ema", "nafnet_medium", HERE / "weights" / "nafnet_medium_v2_ema_best.pth"),
    ]

    tta_modes = ["none", "flip4", "dihedral8"]
    results = []

    print("=" * 80)
    print(f"{'Configuration':<30} | {'dPSNR':<10} | {'dSSIM':<10} | {'Composite':<10} | {'Sec/Img':<8}")
    print("=" * 80)

    for name, arch, path in candidates:
        if not path.exists():
            continue
        model = build_model(arch).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        def predict(x):
            with torch.no_grad():
                return model(x.to(device)).cpu()

        for tta in tta_modes:
            t0 = time.time()
            scores = []
            for gt_p, noisy_p in val_pairs:
                gt, noisy = read_image(gt_p), read_image(noisy_p)
                den = tta_tiled_inference(predict, noisy, tile=384, stride=320, tta_mode=tta)
                scores.append(composite(gt, noisy, den))
            t1 = time.time()
            sec_per_img = (t1 - t0) / len(val_pairs)
            avg_dp = float(np.mean([s["dPSNR"] for s in scores]))
            avg_ds = float(np.mean([s["dSSIM"] for s in scores]))
            avg_score = float(np.mean([s["score"] for s in scores]))

            cfg_name = f"{name}_{tta}"
            entry = {
                "config": cfg_name,
                "weights": str(path.relative_to(ROOT)),
                "arch": arch,
                "tta": tta,
                "dPSNR": avg_dp,
                "dSSIM": avg_ds,
                "composite": avg_score,
                "sec_per_img": sec_per_img
            }
            results.append(entry)
            print(f"{cfg_name:<30} | {avg_dp:+.2f} dB    | {avg_ds:+.3f}     | {avg_score:.4f}     | {sec_per_img:.2f}s")

    print("=" * 80)

    results = sorted(results, key=lambda x: x["composite"], reverse=True)
    best_entry = results[0]
    for r in results:
        r["selected"] = "YES" if r["config"] == best_entry["config"] else "NO"

    # Write final_selection.md
    res_dir = HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    md_path = res_dir / "final_selection.md"

    table_rows = []
    for r in results:
        table_rows.append(f"| `{r['config']}` | `{r['dPSNR']:+.2f} dB` | `{r['dSSIM']:+.3f}` | `{r['composite']:.4f}` | `{r['sec_per_img']:.2f}s` | **{r['selected']}** |")

    table_str = "\n".join(table_rows)

    md_content = f"""# Final Model & Pipeline Selection Report

## Overview
Evaluated candidate models and TTA options on the 29-pair validation set to select the optimal submission pipeline.

## Evaluation Results

| Configuration | Val dPSNR | Val dSSIM | Val Composite | CPU sec/image | Selected |
|---|---|---|---|---|---|
{table_str}

## Selection Rationale
- **Winning Configuration:** `{best_entry['config']}`
- **Model Weights Path:** `{best_entry['weights']}`
- **Validation Composite Score:** `{best_entry['composite']:.4f}` ($\Delta\text{{PSNR}} = {best_entry['dPSNR']:+.2f}\text{{ dB}}$, $\Delta\text{{SSIM}} = {best_entry['dSSIM']:+.3f}$)
- **Score Improvement over Insurance Model:** `{best_entry['composite'] - results[-1]['composite']:+.4f}`
- **GPU Runtime:** `{best_entry['sec_per_img']:.2f}s / image`
"""
    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"\nSaved final selection report to {md_path}")
    print(f"WINNER: {best_entry['config']} (Score: {best_entry['composite']:.4f})")


if __name__ == "__main__":
    import torch
    run_eval()
