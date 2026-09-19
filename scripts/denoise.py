"""Competition entry point. Usage (from REPO_ROOT):
    python scripts/denoise.py --noise_dir x --denoised_dir y
Deep path (NAFNet-lite, tiled) with automatic classical CPU fallback. Fully offline."""
import argparse, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classical import cpu_fallback, estimate_sigma
from utils import derive_output_name, read_image, tiled_inference, write_image


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def resolve_device(name, t):
    return ("cuda" if t.cuda.is_available() else "cpu") if name == "auto" else name


def load_torch_model(weights, device, t):
    from model import build_model
    ckpt = t.load(weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = build_model(cfg.get("model", "nafnet_small"))
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def make_torch_predictor(model, device, t):
    def predict(x):
        with t.no_grad():
            return model(x.to(device)).float().cpu()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--denoised_dir", required=True)
    ap.add_argument("--weights", default=str(HERE / "weights" / "best.pth"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--stride", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--classical-only", action="store_true")
    args = ap.parse_args()

    noise_dir, out_dir = Path(args.noise_dir), Path(args.denoised_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(noise_dir.glob("*.png"))
    if not files:
        sys.exit(f"No PNG files found in {noise_dir}")

    t = _torch()
    predictor = None
    if not args.classical_only:
        if t is None:
            print("[warn] torch not installed -> classical fallback")
        elif Path(args.weights).exists():
            try:
                device = resolve_device(args.device, t)
                model = load_torch_model(args.weights, device, t)
                predictor = make_torch_predictor(model, device, t)
                print(f"[info] deep path on {device}")
            except Exception as e:
                print(f"[warn] weights load failed ({e}) -> classical fallback")
        else:
            print(f"[warn] weights not found at {args.weights} -> classical fallback")

    times = []
    for f in files:
        img = read_image(f)
        sigma = estimate_sigma(img)
        t0 = time.time()
        if predictor is not None:
            try:
                den = tiled_inference(predictor, img, tile=args.tile,
                                      stride=args.stride, batch=args.batch)
            except Exception as e:
                print(f"[warn] deep path failed on {f.name}: {e} -> classical")
                den = cpu_fallback(img, sigma)
        else:
            den = cpu_fallback(img, sigma)
        out_name = derive_output_name(f.name)
        write_image(out_dir / out_name, den)
        dt = time.time() - t0; times.append(dt)
        print(f"{f.name} -> {out_name} | sigma={sigma:.4f} | {dt:.1f}s")
    print(f"[done] {len(files)} images, avg {np.mean(times):.1f}s/image -> {out_dir}")


if __name__ == "__main__":
    main()
