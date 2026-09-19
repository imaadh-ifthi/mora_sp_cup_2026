"""One-command pipeline (run from REPO_ROOT): analyze -> train -> denoise -> verify -> zip.
    python scripts/run_all.py --team fit [--skip-train] [--skip-analysis]"""
import argparse, hashlib, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
# AGENT: after Phase 0 step 6, fill this with the official evaluator command.
EVAL_CMD = [PY, "evaluation/evaluate.py", "--noisy_dir", "competition_data/public/noisy", "--pred_dir", "competition_data/submissions/denoised", "--gt_dir", "competition_data/public/ground_truth"]


def run(cmd):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="fit")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    pub = ROOT / "competition_data" / "public"
    sub = ROOT / "competition_data" / "submissions"

    if not args.skip_analysis:
        run([PY, HERE / "noise_analysis.py", "--gt_dir", pub / "ground_truth",
             "--noisy_dir", pub / "noisy", "--num", "60",
             "--out", str(HERE / "analysis.json")])
    if not args.skip_train:
        run([PY, HERE / "train.py", "--config", str(HERE / "config.yaml")])
    run([PY, HERE / "denoise.py", "--noise_dir", sub / "noisy",
         "--denoised_dir", sub / "denoised"])

    den = sorted((sub / "denoised").glob("*.png"))
    expected = {f"{i}.png" for i in range(461, 481)}
    got = {p.name for p in den}
    missing = expected - got
    assert not missing, f"missing outputs: {sorted(missing)}"
    zpath = ROOT / f"{args.team}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(den, key=lambda p: int(p.stem)):
            if p.name in expected:
                z.write(p, p.name)
    print(f"[ok] wrote {zpath} with {len(expected)} images")

    w = HERE / "weights" / "best.pth"
    if w.exists():
        print(f"[weights] {w}\n[weights] SHA-256: {sha256(w)}")
    if EVAL_CMD:
        run(EVAL_CMD)
    else:
        print("[info] EVAL_CMD not set — run evaluation/evaluate.py manually for official self-eval.")


if __name__ == "__main__":
    main()
