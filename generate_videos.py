# generate_videos.py — WAN 2.1 T2V gen + optional LoRA (OOM-hardened)
import os, argparse, yaml, warnings, gc
from pathlib import Path

# WAN needs ftfy for text cleanup
try:
    import ftfy  # noqa: F401
except Exception as e:
    raise RuntimeError("WAN pipeline requires `ftfy`.\nIn Colab:  pip -q install ftfy") from e

# Try to reduce CUDA fragmentation early
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def round_wan_frames(n: int) -> int:
    # WAN requires (num_frames - 1) % 4 == 0
    if (n - 1) % 4 == 0:
        return n
    r = 4 * round((n - 1) / 4) + 1
    return int(max(5, r))

def to_hwc_uint8(frame) -> "np.ndarray":
    if isinstance(frame, torch.Tensor):
        fr = frame.detach().clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        import numpy as np
        fr = frame
        if fr.dtype != np.uint8:
            fr = fr.clip(0, 255).astype("uint8")
    return fr

def maybe_load_lora(pipe, lora_path: str):
    if not lora_path:
        return
    if not os.path.isdir(lora_path):
        print(f"[LoRA] WARNING: '{lora_path}' is not a directory saved via save_attn_procs(). Continuing base-only.")
        return
    try:
        pipe.transformer.load_attn_procs(lora_path)
        print(f"[LoRA] Loaded attention processors from: {lora_path}")
    except Exception as e:
        print(f"[LoRA] WARNING: failed to load from '{lora_path}': {e}. Continuing base-only.")

def enable_memory_savers(pipe):
    # These exist in Diffusers and help a lot during WAN decode
    try: pipe.enable_attention_slicing("max")
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    # xFormers if Colab already has it
    try:
        import xformers  # noqa: F401
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    except Exception:
        pass
    # TF32 matmul is okay for speed on Ampere
    try: torch.backends.cuda.matmul.allow_tf32 = True
    except Exception: pass

def autocast_ctx(device, dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)

def run_one(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype):
    # Prefer MAGMA to dodge occasional cuSOLVER hiccups
    prev_backend = None
    if torch.cuda.is_available():
        try: prev_backend = torch.backends.cuda.preferred_linalg_library()
        except Exception: pass
        try: torch.backends.cuda.preferred_linalg_library("magma")
        except Exception: pass

    try:
        with torch.inference_mode(), autocast_ctx(device, dtype):
            return pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_frames=frames,
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
    except torch.cuda.OutOfMemoryError:
        # Auto-retry smaller (halved res + fewer frames/steps), still valid for WAN
        torch.cuda.empty_cache()
        smaller = max(256, (size // 2) // 8 * 8)   # keep multiple of 8 just in case
        fewer_frames = round_wan_frames(max(9, frames - 4))
        fewer_steps  = max(12, steps // 2)
        print(f"[OOM] Retrying with size {size}->{smaller}, frames {frames}->{fewer_frames}, steps {steps}->{fewer_steps}")
        with torch.inference_mode(), autocast_ctx(device, dtype):
            return pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_frames=fewer_frames,
                height=smaller,
                width=smaller,
                num_inference_steps=fewer_steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
    finally:
        # Restore previous linalg backend pref if known
        if prev_backend is not None:
            try: torch.backends.cuda.preferred_linalg_library(prev_backend)
            except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg)

    model_name = cfg["model"].get("name", "wan-2.1-1.3b")
    base_ckpt  = cfg["model"]["base_ckpt"]
    lora_path  = cfg["model"].get("lora_path", "")

    seed       = int(cfg.get("sampler", {}).get("seed", 42))
    frames     = int(cfg["sampler"]["frames"])
    fps        = int(cfg["sampler"]["fps"])
    size       = int(cfg["sampler"]["size"])
    steps      = int(cfg["sampler"]["steps"])
    cfg_scale  = float(cfg["sampler"]["cfg_scale"])
    neg_prompt = cfg["sampler"].get("negative_prompt", None)

    classes    = cfg["dataset"]["classes"]
    per_class  = int(cfg["dataset"]["per_class"])
    out_dir    = Path(cfg["dataset"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # WAN requires (frames-1) % 4 == 0
    new_frames = round_wan_frames(frames)
    if new_frames != frames:
        print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
        frames = new_frames

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,   # WAN will ignore if unsupported; harmless
        variant=None,
    ).to(device)

    enable_memory_savers(pipe)
    maybe_load_lora(pipe, lora_path)

    # Generator on device for deterministic sampling
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating videos …")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"

            result = run_one(
                pipe=pipe,
                prompt=prompt,
                neg=neg_prompt,
                frames=frames,
                size=size,
                steps=steps,
                cfg_scale=cfg_scale,
                generator=generator,
                device=device,
                dtype=dtype,
            )

            frames_rgb = [to_hwc_uint8(fr) for fr in result.frames]
            path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(path, frames_rgb, fps=fps)

            # Free everything before the next sample
            del result, frames_rgb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("✅ All videos generated successfully.")

if __name__ == "__main__":
    main()
