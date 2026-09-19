"""Training loop tuned for RTX 3050 (AMP + small tiles). Validation uses the exact
official composite formula on a ~30-pair hold-out. Paths resolve relative to REPO_ROOT
so this works no matter which folder you launch it from."""
import argparse, random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import PairDataset, list_pairs
from metrics import SSIMLoss, composite
from model import build_model
from noise_model import SyntheticNoiseModel
from utils import read_image, tiled_inference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_predictor(model, device):
    def predict(x):
        with torch.no_grad():
            return model(x.to(device)).cpu()
    return predict


def validate(model, pairs, device, tile=384, stride=320, limit=None):
    model.eval()
    pred = make_predictor(model, device)
    rows = []
    for gt_p, noisy_p in (pairs if limit is None else pairs[:limit]):
        gt, noisy = read_image(gt_p), read_image(noisy_p)
        den = tiled_inference(pred, noisy, tile=tile, stride=stride)
        rows.append(composite(gt, noisy, den))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("dPSNR", "dSSIM", "score")}
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} model={cfg['model']}")

    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = ROOT / root          # resolve relative to REPO_ROOT
    pairs = list_pairs(root / "ground_truth", root / "noisy")
    assert pairs, "No pairs found — check data_root in config.yaml"
    idx = list(range(len(pairs))); random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(pairs) * cfg["val_ratio"]))
    val_pairs = [pairs[i] for i in idx[:n_val]]
    train_pairs = [pairs[i] for i in idx[n_val:]]
    print(f"[train] pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")

    synth = SyntheticNoiseModel(**cfg["noise"])
    ds = PairDataset(train_pairs, crop=cfg["tile"], augment=True,
                     synth=synth, synth_prob=cfg["synth_prob"], seed=seed)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True)

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ssim_loss = SSIMLoss().to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    wdir = ROOT / cfg["weights_dir"]; wdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for noisy, gt in loader:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(noisy)
                    loss = (pred - gt).abs().mean() + cfg["ssim_weight"] * ssim_loss(pred, gt)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tot += loss.item(); n += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print("[warn] OOM on a batch — skipped. Consider smaller tile/batch.")
                    continue
                raise
        sched.step()
        print(f"[epoch {epoch:3d}] loss={tot / max(n, 1):.4f} lr={opt.param_groups[0]['lr']:.2e}")

        if epoch % cfg.get("val_every", 5) == 0 or epoch == cfg["epochs"]:
            agg, _ = validate(model, val_pairs, device, limit=cfg.get("val_limit", 15))
            print(f"[val] dPSNR={agg['dPSNR']:+.2f} dB dSSIM={agg['dSSIM']:+.3f} "
                  f"composite={agg['score']:.4f} (baseline=0.26)")
            if agg["score"] > best:
                best = agg["score"]
                torch.save({"model": model.state_dict(), "config": {"model": cfg["model"]},
                            "epoch": epoch, "score": best}, wdir / "best.pth")
                print(f"[val] saved best.pth (score={best:.4f})")
    print(f"[train] done. best validation composite={best:.4f}")
    assert best > 0.26, "Did not beat baseline 0.26 — do not submit; retrain with more epochs."


if __name__ == "__main__":
    main()
