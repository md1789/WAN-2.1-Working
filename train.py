# train.py
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, textwrap
from pathlib import Path
from typing import Optional

# -----------------------
# small paths helper (you already had this)
# -----------------------
class WanModelPaths:
    def __init__(self,
                 base_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                 output_dir: str = "outputs/wan_lora",
                 cache_dir: Optional[str] = None):
        self.base_model_id = base_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

def get_default_paths():
    return WanModelPaths()

# -----------------------
# utility
# -----------------------
def echo_box(msg: str) -> None:
    bar = "═" * 60
    print(f"\n\u2554{bar}\u2557")
    for line in msg.splitlines():
        print("\u2551 " + line.ljust(60) + " \u2551")
    print(f"\u255A{bar}\u255D\n")

def write_placeholder_config(cfg_path: Path,
                             base_model_id: str,
                             data_dir: str,
                             out_dir: str) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # This mirrors the WAN video-LoRA style configs found in community trainers.
    # Treat it as a starting point—edit paths/hparams as needed.
    cfg = f"""\
# Auto-generated starter config for WAN 2.1 LoRA training
# Adjust these to your setup. Values here are safe-ish defaults to get you moving.

[project]
name = "wan21_lora"
output_root = "{out_dir}"

[model]
# You can point to a local folder with weights or a hub id
pretrained_model_name_or_path = "{base_model_id}"
# if you downloaded weights locally, set e.g.:
# pretrained_model_name_or_path = "weights/wan21"

[precision]
dtype = "bf16"           # "fp16" if your GPU prefers it
grad_checkpointing = true

[optimizer]
lr = 1e-4
betas = [0.9, 0.999]
weight_decay = 0.0

[scheduler]
type = "cosine"
warmup_steps = 100

[train]
seed = 42
max_steps = 1000
save_every = 200
log_every = 50
train_batch_size_per_gpu = 1
num_workers = 4

[data]
type = "video_folder"
root = "{data_dir}"
# For T2V/I2V LoRA, trainer will parse videos (and optional .txt sidecar prompts)
# You can also provide a CSV/JSON dataset in some trainers—this keeps it simple.

[video]
num_frames = 16
fps = 15
# Height/width should match what the base model expects; adjust to your data.
height = 480
width  = 848

[lora]
enable = true
rank = 16
alpha = 32
# target_modules are backend-specific in each trainer; using a broad default here.
# If your trainer exposes a default list, you can delete this entry.
target_modules = ["q_proj","k_proj","v_proj","out_proj","to_q","to_k","to_v","to_out"]

[logging]
backend = "tensorboard"
"""
    cfg_path.write_text(cfg, encoding="utf-8")

def main() -> int:
    default = get_default_paths()

    ap = argparse.ArgumentParser(
        description="WAN 2.1 LoRA launcher (thin wrapper around diffusion-pipe)."
    )
    ap.add_argument("--dp_path", default="third_party/diffusion-pipe",
                    help="Path to the diffusion-pipe repo (added as git submodule).")
    ap.add_argument("--config", "-c", default="configs/my_wan21.toml",
                    help="TOML config for the trainer. Will be created if missing.")
    ap.add_argument("--data_dir", default="data",
                    help="Folder containing your training data (videos + optional .txt captions).")
    ap.add_argument("--output_dir", default=str(default.output_dir),
                    help="Where checkpoints/logs will go.")
    ap.add_argument("--base_model_id", default=default.base_model_id,
                    help="Hub ID or local path for WAN 2.1 base weights.")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--use_deepspeed", action="store_true",
                    help="Launch via deepspeed. Recommended on Linux/WSL2.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the command without executing.")
    ap.add_argument("--force_cfg", action="store_true",
                    help="Overwrite existing config with a fresh template.")
    args = ap.parse_args()

    repo = Path(args.dp_path).resolve()
    cfg  = Path(args.config).resolve()
    data = Path(args.data_dir).resolve()
    out  = Path(args.output_dir).resolve()

    # Basic sanity checks
    if not repo.exists():
        echo_box(textwrap.dedent(f"""
        Missing trainer repo at: {repo}
        Tip:
          git submodule add https://github.com/tdrussell/diffusion-pipe third_party/diffusion-pipe
          git submodule update --init --recursive
        """))
        return 2

    train_py = repo / "train.py"
    if not train_py.exists():
        echo_box(f"Could not find trainer script at: {train_py}\nDid submodules finish cloning?")
        return 2

    # Create a config if needed
    if args.force_cfg or not cfg.exists():
        # Try to copy an example if one exists; otherwise write a placeholder
        example = repo / "examples" / "wan_video.toml"
        if example.exists() and not args.force_cfg:
            cfg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(example, cfg)
            # Soft-edit a few obvious fields if present (best-effort)
            text = cfg.read_text(encoding="utf-8")
            text = text.replace("DATA_ROOT_HERE", str(data))
            text = text.replace("OUTPUT_ROOT_HERE", str(out))
            text = text.replace("PRETRAINED_MODEL_HERE", args.base_model_id)
            cfg.write_text(text, encoding="utf-8")
            echo_box(f"Config created from example: {cfg}")
        else:
            write_placeholder_config(cfg, args.base_model_id, str(data), str(out))
            echo_box(f"Config created: {cfg}\n(Adjust it as needed before training.)")

    # Environment tweaks that help on single-GPU boxes
    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    # Build the launch command
    if args.use_deepspeed:
        # deepspeed --num_gpus=1 third_party/diffusion-pipe/train.py --deepspeed --config cfg.toml
        cmd = ["deepspeed", f"--num_gpus={args.num_gpus}", str(train_py), "--deepspeed", "--config", str(cfg)]
    else:
        # python third_party/diffusion-pipe/train.py --config cfg.toml
        cmd = [sys.executable, str(train_py), "--config", str(cfg)]

    echo_box("Launching trainer with command:\n" + " ".join(cmd))

    if args.dry_run:
        echo_box("Dry run complete. No processes started.")
        return 0

    proc = subprocess.run(cmd, env=env)
    return proc.returncode

if __name__ == "__main__":
    raise SystemExit(main())
