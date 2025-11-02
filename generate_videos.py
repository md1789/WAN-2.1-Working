# generate_videos.py — WAN 2.1 T2V generation + optional LoRA
import os, argparse, yaml, warnings
from pathlib import Path

import torch
import imageio.v2 as imageio
from tqdm import tqdm
from diffusers import DiffusionPipeline

warnings.filterwarnings("ignore", category=UserWarning)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_dtype(cfg):
    bf16 = bool(cfg.get("compute", {}).get("bf16", False))
    return torch.bfloat16 if (bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg)

    model_name = cfg["model"]["name"]
    base_ckpt  = cfg["model"]["base_ckpt"]
    lora_path  = cfg["model"].get("lora_path", "")
    seed       = int(cfg.get("sampler", {}).get("seed", 42))

    frames     = int(cfg["sampler"]["frames"])
    fps        = int(cfg["sampler"]["fps"])
    size       = int(cfg["sampler"]["size"])
    steps      = int(cfg["sampler"]["steps"])
    cfg_scale  = float(cfg["sampler"]["cfg_scale"])

    classes    = cfg["dataset"]["classes"]
    per_class  = int(cfg["dataset"]["per_class"])
    out_dir    = Path(cfg["dataset"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,            # new arg name per recent diffusers
        variant=None,
    ).to(device)

    # Load LoRA attention processors if provided
    if lora_path and os.path.exists(lora_path):
        try:
            pipe.transformer.load_attn_procs(lora_path)
            print(f"Loaded LoRA attention processors from: {lora_path}")
        except Exception as e:
            print(f"WARNING: Failed to load LoRA from '{lora_path}': {e}")
    else:
        if lora_path:
            print(f"WARNING: LoRA path not found: {lora_path} — proceeding with base model.")

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating videos …")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    num_frames=frames,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                )

            # result.frames: list[np.ndarray(H,W,3)] or tensor; ensure HWC uint8
            frames_rgb = []
            for fr in result.frames:
                fr = fr if isinstance(fr, torch.Tensor) else torch.from_numpy(fr)
                fr = fr.clamp(0, 255).to(torch.uint8).cpu().numpy()
                frames_rgb.append(fr)

            path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(path, frames_rgb, fps=fps)
    print("✅ All videos generated successfully.")

if __name__ == "__main__":
    main()
