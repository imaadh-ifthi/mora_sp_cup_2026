"""Dataset diagnostics for the report: darkening check, severity stats, var-mean fit."""
import argparse, json
from pathlib import Path
import numpy as np
from classical import estimate_sigma
from dataset import list_pairs
from metrics import psnr
from utils import read_image


def var_mean_fit(pairs, patches_per_image=20, patch=64, seed=0):
    rng = np.random.default_rng(seed)
    mus, vars_ = [], []
    for gt_p, noisy_p in pairs:
        gt, noisy = read_image(gt_p), read_image(noisy_p)
        H, W, _ = gt.shape
        for _ in range(patches_per_image):
            top = int(rng.integers(0, H - patch)); left = int(rng.integers(0, W - patch))
            mus.append(float(gt[top:top+patch, left:left+patch].mean()))
            vars_.append(float((noisy - gt)[top:top+patch, left:left+patch].var()))
    mus, vars_ = np.array(mus), np.array(vars_)
    A = np.vstack([mus, np.ones_like(mus)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, vars_, rcond=None)
    return {"slope": float(slope), "intercept": float(intercept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--noisy_dir", required=True)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--out", default="scripts/analysis.json")
    args = ap.parse_args()
    pairs = list_pairs(args.gt_dir, args.noisy_dir)
    print(f"found {len(pairs)} pairs")
    step = max(1, len(pairs) // args.num)
    sample = pairs[::step][:args.num]
    rows = []
    for gt_p, noisy_p in sample:
        gt, noisy = read_image(gt_p), read_image(noisy_p)
        rows.append({"name": Path(noisy_p).name, "sigma_hat": estimate_sigma(noisy),
                     "brightness_ratio": float(noisy.mean() / max(gt.mean(), 1e-6)),
                     "psnr_noisy": psnr(gt, noisy)})
    sig = np.array([r["sigma_hat"] for r in rows]); rat = np.array([r["brightness_ratio"] for r in rows])
    pn = np.array([r["psnr_noisy"] for r in rows])
    summary = {"n_sampled": len(rows),
               "sigma": {"min": float(sig.min()), "median": float(np.median(sig)), "max": float(sig.max())},
               "brightness_ratio": {"min": float(rat.min()), "median": float(np.median(rat)), "max": float(rat.max())},
               "darkening_detected": bool(np.median(rat) < 0.95),
               "psnr_noisy": {"min": float(pn.min()), "median": float(np.median(pn)), "max": float(pn.max())},
               "var_mean_fit": var_mean_fit(sample), "per_image": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    print("NOTE: noisy images DARKER than GT" if summary["darkening_detected"]
          else "NOTE: no global darkening; degradation is mostly additive noise")


if __name__ == "__main__":
    main()
