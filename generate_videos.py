# generate_videos.py — WAN 2.1 LoRA video generation
import os, argparse, yaml
from pathlib import Path
from tqdm import tqdm
import torch
import imageio.v2 as imageio
from diffusers import DiffusionPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model"]["name"]
    base_ckpt  = cfg["model"]["base_ckpt"]
    lora_path  = cfg["model"]["lora_path"]
    frames     = int(cfg["sampler"]["frames"])
    fps        = int(cfg["sampler"]["fps"])
    size       = int(cfg["sampler"]["size"])
    steps      = int(cfg["sampler"]["steps"])
    cfg_scale  = float(cfg["sampler"]["cfg_scale"])

    classes    = cfg["dataset"]["classes"]
    per_class  = int(cfg["dataset"]["per_class"])
    out_dir    = Path(cfg["dataset"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA...")

    base_ckpt = cfg["model"].get("base_ckpt", None)
    print(f"DEBUG: Attempting to load base_ckpt = {cfg['model'].get('base_ckpt')}")
    if not base_ckpt:
        raise ValueError("Missing 'base_ckpt' in config['model'] — check your YAML!")

    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        torch_dtype=torch.float16,
    ).to(device)


    # Load LoRA adapters
    if os.path.exists(lora_path):
        pipe.unet.load_attn_procs(lora_path)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        print(f"LoRA weights not found at {lora_path}; using base model only.")

    print("Generating videos...")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(per_class), desc=f"{cls:>10}"):
            prompt = f"a person performing {cls}"
            with torch.no_grad(), torch.cuda.amp.autocast():
                result = pipe(
                    prompt,
                    num_frames=frames,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                )

            # Save frames as .mp4
            path = cdir / f"{cls}_{i:02d}.mp4"
            frames_rgb = [frame for frame in result.frames]
            imageio.mimwrite(path, frames_rgb, fps=fps)
    print("All videos generated successfully.")

if __name__ == "__main__":
    main()
