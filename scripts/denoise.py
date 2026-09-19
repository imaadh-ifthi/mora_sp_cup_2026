"""Competition entry point. Usage (from REPO_ROOT):
    python scripts/denoise.py --noise_dir x --denoised_dir y
Deep path (NAFNet-lite + TTA, tiled) with automatic classical CPU fallback. Fully offline."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classical import cpu_fallback, estimate_sigma
from tta import tta_tiled_inference
from utils import derive_output_name, read_image, tiled_inference, write_image


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def resolve_device(name, t):
    if name == "auto":
        return "cuda" if (t is not None and t.cuda.is_available()) else "cpu"
    return name


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
    ap.add_argument("--manifest", default=str(HERE / "weights" / "manifest.json"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--tta", default=None, choices=[None, "none", "flip2", "flip4", "dihedral8"])
    ap.add_argument("--classical-only", action="store_true")
    args = ap.parse_args()

    noise_dir, out_dir = Path(args.noise_dir), Path(args.denoised_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(noise_dir.glob("*.png"))
    if not files:
        sys.exit(f"No PNG files found in {noise_dir}")

    # Manifest resolution
    tile = args.tile
    stride = args.stride
    tta_mode = args.tta
    weights_path = args.weights

    manifest_p = Path(args.manifest)
    if manifest_p.exists():
        try:
            m_data = json.load(open(manifest_p))
            inf_cfg = m_data.get("inference", {})
            if tile is None:
                tile = inf_cfg.get("tile", 384)
            if stride is None:
                stride = inf_cfg.get("stride", 320)
            if tta_mode is None:
                tta_mode = inf_cfg.get("tta", "dihedral8")
            if m_data.get("models") and len(m_data["models"]) > 0:
                rel_w = m_data["models"][0].get("weights")
                if rel_w:
                    cand_w = HERE / ".." / rel_w if not Path(rel_w).is_absolute() else Path(rel_w)
                    if cand_w.exists():
                        weights_path = str(cand_w.resolve())
        except Exception as e:
            print(f"[warn] manifest parse error ({e}) -> using defaults")

    if tile is None:
        tile = 384
    if stride is None:
        stride = 320
    if tta_mode is None:
        tta_mode = "dihedral8"

    t = _torch()
    predictor = None
    if not args.classical_only:
        if t is None:
            print("[warn] torch not installed -> classical fallback")
        elif Path(weights_path).exists():
            try:
                device = resolve_device(args.device, t)
                model = load_torch_model(weights_path, device, t)
                predictor = make_torch_predictor(model, device, t)
                print(f"[info] deep path on {device} (weights={Path(weights_path).name}, tile={tile}, stride={stride}, tta={tta_mode})")
            except Exception as e:
                print(f"[warn] weights load failed ({e}) -> classical fallback")
        else:
            print(f"[warn] weights not found at {weights_path} -> classical fallback")

    times = []
    for f in files:
        img = read_image(f)
        sigma = estimate_sigma(img)
        t0 = time.time()
        if predictor is not None:
            try:
                den = tta_tiled_inference(predictor, img, tile=tile, stride=stride, tta_mode=tta_mode)
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
