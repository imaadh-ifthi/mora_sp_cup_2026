"""Inference benchmarking script.
Evaluates tile/stride combinations and TTA modes (none, flip4, dihedral8) on validation images.
Reports dPSNR, dSSIM, composite score, GPU VRAM usage, and CPU/GPU time per image.
"""
import sys, time, torch, yaml, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from dataset import list_pairs
from metrics import composite
from model import build_model
from tta import tta_tiled_inference
from utils import read_image


def benchmark():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    seed = cfg["seed"]
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = list_pairs(ROOT / cfg["data_root"] / "ground_truth", ROOT / cfg["data_root"] / "noisy")
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(pairs) * cfg["val_ratio"]))
    val_pairs = [pairs[i] for i in idx[:n_val]][:5]  # benchmark on 5 val pairs for fast feedback

    model_path = HERE / "weights" / "best.pth"
    if not model_path.exists():
        model_path = HERE / "weights" / "insurance_best.pth"

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def predict(x):
        with torch.no_grad():
            return model(x.to(device)).cpu()

    benchmarks = [
        {"tile": 384, "stride": 320, "tta": "none"},
        {"tile": 384, "stride": 320, "tta": "flip4"},
        {"tile": 384, "stride": 320, "tta": "dihedral8"},
        {"tile": 448, "stride": 384, "tta": "none"},
        {"tile": 448, "stride": 384, "tta": "flip4"},
    ]

    print("=" * 65, flush=True)
    print(f"{'Tile/Stride':<12} | {'TTA':<10} | {'dPSNR':<8} | {'dSSIM':<8} | {'Score':<8} | {'Sec/Img':<8}", flush=True)
    print("=" * 65, flush=True)

    results = []
    for b in benchmarks:
        tile, stride, tta_mode = b["tile"], b["stride"], b["tta"]
        t0 = time.time()
        scores = []
        for gt_p, noisy_p in val_pairs:
            gt, noisy = read_image(gt_p), read_image(noisy_p)
            den = tta_tiled_inference(predict, noisy, tile=tile, stride=stride, tta_mode=tta_mode)
            res = composite(gt, noisy, den)
            scores.append(res)
        t1 = time.time()
        sec_per_img = (t1 - t0) / len(val_pairs)
        avg_dp = float(np.mean([s["dPSNR"] for s in scores]))
        avg_ds = float(np.mean([s["dSSIM"] for s in scores]))
        avg_score = float(np.mean([s["score"] for s in scores]))

        print(f"{tile}/{stride:<7} | {tta_mode:<10} | {avg_dp:+.2f} dB | {avg_ds:+.3f} | {avg_score:.4f}  | {sec_per_img:.2f}s", flush=True)
        results.append({
            "tile": tile, "stride": stride, "tta": tta_mode,
            "dPSNR": avg_dp, "dSSIM": avg_ds, "score": avg_score, "sec_per_img": sec_per_img
        })

    print("=" * 65, flush=True)


if __name__ == "__main__":
    benchmark()
