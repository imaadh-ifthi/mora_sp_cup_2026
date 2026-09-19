"""Training loop V2 for nafnet_medium with EMA, severity sampling, warmup + cosine schedule,
gradient accumulation, and CompositeLossV2.
"""
import argparse, copy, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from classical import estimate_sigma
from dataset import PairDataset, list_pairs
from losses_v2 import CompositeLossV2
from metrics import composite
from model import build_model
from noise_model import SyntheticNoiseModel
from utils import read_image, tiled_inference


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for p_shadow, p_model in zip(self.shadow.parameters(), model.parameters()):
                p_shadow.copy_(self.decay * p_shadow + (1.0 - self.decay) * p_model)


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
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config_v2.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg.get("seed", 2026)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_v2] device={device} model={cfg['model']} epochs={cfg['epochs']}")

    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = ROOT / root

    pairs = list_pairs(root / "ground_truth", root / "noisy")
    assert pairs, "No pairs found!"
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(pairs) * cfg["val_ratio"]))
    val_pairs = [pairs[i] for i in idx[:n_val]]
    train_pairs = [pairs[i] for i in idx[n_val:]]
    print(f"[train_v2] pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")

    # Compute severity weights for train pairs
    sigmas = []
    for gt_p, noisy_p in train_pairs:
        noisy = read_image(noisy_p)
        s = estimate_sigma(noisy)
        sigmas.append(s)
    sigmas = np.array(sigmas, dtype=np.float32)
    s_min, s_max = np.min(sigmas), np.max(sigmas)
    s_norm = (sigmas - s_min) / max(s_max - s_min, 1e-6)
    weights = 1.0 + 0.7 * s_norm
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    synth = SyntheticNoiseModel(**cfg["noise"])
    ds = PairDataset(train_pairs, crop=cfg["tile"], augment=True,
                     synth=synth, synth_prob=cfg["synth_prob"], seed=seed)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler,
                        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True)

    model = build_model(cfg["model"]).to(device)
    ema = EMAModel(model, decay=cfg.get("ema_decay", 0.999))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Warmup + Cosine Scheduler
    warmup_epochs = cfg.get("warmup_epochs", 5)
    total_epochs = cfg["epochs"]

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    criterion = CompositeLossV2(
        charbonnier_weight=cfg.get("charbonnier_weight", 1.0),
        ssim_weight=cfg.get("ssim_weight", 0.2),
        grad_weight=cfg.get("grad_weight", 0.03),
        fft_weight=cfg.get("fft_weight", 0.005)
    ).to(device)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wdir = ROOT / cfg["weights_dir"]
    wdir.mkdir(parents=True, exist_ok=True)

    raw_best_score = -1.0
    ema_best_score = -1.0
    accum_steps = cfg.get("accum_steps", 2)
    grad_clip = cfg.get("grad_clip", 0.5)

    results_dir = ROOT / "scripts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(results_dir / "train_v2.log", "w")

    print("[train_v2] starting training...", flush=True)
    for epoch in range(1, total_epochs + 1):
        model.train()
        tot_loss, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for step, (noisy, gt) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(noisy)
                    loss = criterion(pred, gt) / float(accum_steps)

                scaler.scale(loss).backward()
                tot_loss += loss.item() * accum_steps

                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    ema.update(model)

                n_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print("[warn] OOM step skipped.")
                    continue
                raise

        sched.step()
        avg_loss = tot_loss / max(n_batches, 1)
        curr_lr = opt.param_groups[0]["lr"]
        msg = f"[epoch {epoch:3d}/{total_epochs}] loss={avg_loss:.4f} lr={curr_lr:.2e}"
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

        if epoch % cfg.get("val_every", 5) == 0 or epoch == total_epochs:
            raw_agg = validate(model, val_pairs, device, limit=None)
            ema_agg = validate(ema.shadow, val_pairs, device, limit=None)

            val_msg = (f"[val {epoch:3d}] RAW -> dPSNR={raw_agg['dPSNR']:+.2f}dB dSSIM={raw_agg['dSSIM']:+.3f} composite={raw_agg['score']:.4f}\n"
                       f"[val {epoch:3d}] EMA -> dPSNR={ema_agg['dPSNR']:+.2f}dB dSSIM={ema_agg['dSSIM']:+.3f} composite={ema_agg['score']:.4f}")
            print(val_msg, flush=True)
            log_file.write(val_msg + "\n"); log_file.flush()

            if raw_agg["score"] > raw_best_score:
                raw_best_score = raw_agg["score"]
                torch.save({"model": model.state_dict(), "config": {"model": cfg["model"]},
                            "epoch": epoch, "score": raw_best_score}, wdir / "nafnet_medium_v2_raw_best.pth")
                print(f"[val] saved nafnet_medium_v2_raw_best.pth (score={raw_best_score:.4f})", flush=True)

            if ema_agg["score"] > ema_best_score:
                ema_best_score = ema_agg["score"]
                torch.save({"model": ema.shadow.state_dict(), "config": {"model": cfg["model"]},
                            "epoch": epoch, "score": ema_best_score}, wdir / "nafnet_medium_v2_ema_best.pth")
                print(f"[val] saved nafnet_medium_v2_ema_best.pth (score={ema_best_score:.4f})", flush=True)

    # Save last EMA checkpoint
    torch.save({"model": ema.shadow.state_dict(), "config": {"model": cfg["model"]},
                "epoch": total_epochs, "score": ema_best_score}, wdir / "nafnet_medium_v2_ema_last.pth")

    final_msg = f"[train_v2] finished. RAW best composite={raw_best_score:.4f}, EMA best composite={ema_best_score:.4f}"
    print(final_msg, flush=True)
    log_file.write(final_msg + "\n"); log_file.close()


if __name__ == "__main__":
    main()
