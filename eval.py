# eval.py
from __future__ import annotations
import argparse, os, sys, subprocess, textwrap
from pathlib import Path
from typing import Iterable, Optional

# --------- helpers ---------
CANDIDATE_ENTRYPOINTS: tuple[str, ...] = (
    "sample.py",
    "inference.py",
    "infer.py",
    "sample_video.py",
    "generate.py",
)

LORA_EXTS = (".safetensors", ".pt", ".bin")

def echo_box(msg: str) -> None:
    bar = "═" * 70
    print(f"\n\u2554{bar}\u2557")
    for line in msg.splitlines():
        print("\u2551 " + line.ljust(70) + " \u2551")
    print(f"\u255A{bar}\u255D\n")

def newest(paths: Iterable[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def find_lora_file(root: Path) -> Optional[Path]:
    # common places: outputs/**/lora/*.safetensors, outputs/**/adapter*.safetensors, etc.
    candidates: list[Path] = []
    for ext in LORA_EXTS:
        candidates += list(root.rglob(f"*lora*{ext}"))
        candidates += list(root.rglob(f"*adapter*{ext}"))
        candidates += list(root.rglob(f"*LoRA*{ext}"))
    return newest(candidates)

def find_entrypoint(dp_path: Path) -> Optional[Path]:
    for name in CANDIDATE_ENTRYPOINTS:
        p = dp_path / name
        if p.exists():
            return p
        # some repos put entrypoints under a subdir
        for sub in ("scripts", "examples", "tools"):
            q = dp_path / sub / name
            if q.exists():
                return q
    return None

# --------- main ---------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a quick WAN 2.1 LoRA sample via diffusion-pipe."
    )
    ap.add_argument("--dp_path", default="third_party/diffusion-pipe",
                    help="Path to diffusion-pipe repo.")
    ap.add_argument("--config", "-c", default="configs/my_wan21.toml",
                    help="Trainer TOML config (used by most sample scripts).")
    ap.add_argument("--outputs_root", default="outputs",
                    help="Root folder where LoRA checkpoints were saved.")
    ap.add_argument("--lora_path", default=None,
                    help="Explicit path to a LoRA file (.safetensors/.pt). If omitted, newest under outputs_root is used.")
    ap.add_argument("--prompt", required=False, default="A serene ocean at golden hour, gentle camera dolly forward",
                    help="Text prompt for sampling.")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=848)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--out", default="outputs/eval/sample.mp4",
                    help="Output video path.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dp = Path(args.dp_path).resolve()
    if not dp.exists():
        echo_box(textwrap.dedent(f"""
        diffusion-pipe not found at: {dp}
        Tip:
          git submodule add https://github.com/tdrussell/diffusion-pipe third_party/diffusion-pipe
          git submodule update --init --recursive
        """))
        return 2

    entry = find_entrypoint(dp)
    if not entry:
        echo_box("Could not locate a sample/inference entrypoint in diffusion-pipe.\n"
                 f"Tried: {', '.join(CANDIDATE_ENTRYPOINTS)}")
        return 2

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lora_path = Path(args.lora_path).resolve() if args.lora_path else None
    if lora_path is None:
        newest_lora = find_lora_file(Path(args.outputs_root).resolve())
        if not newest_lora:
            echo_box(textwrap.dedent(f"""
            No LoRA checkpoint found under: {Path(args.outputs_root).resolve()}
            Expected files like *lora*.safetensors or *adapter*.safetensors.
            You can pass --lora_path /path/to/your_lora.safetensors explicitly.
            """))
            return 2
        lora_path = newest_lora

    # Build a "best guess" command; most diffusion-pipe variants accept a similar API.
    # If your version uses different flags, the script will fail with its help message.
    cmd = [
        sys.executable, str(entry),
        "--config", str(Path(args.config).resolve()),
        "--prompt", args.prompt,
        "--lora_path", str(lora_path),
        "--out", str(out_path),
        "--num_frames", str(args.num_frames),
        "--height", str(args.height),
        "--width",  str(args.width),
        "--fps", str(args.fps),
        "--steps", str(args.steps),
        "--seed",  str(args.seed),
        "--guidance", str(args.guidance),
    ]

    echo_box("Eval launching with:\n" + " ".join(cmd) + f"\n\nLoRA: {lora_path}")

    if args.dry_run:
        echo_box("Dry run only. No sampling executed.")
        return 0

    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    return subprocess.call(cmd, env=env)

if __name__ == "__main__":
    raise SystemExit(main())
