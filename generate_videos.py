# generate_videos.py — WAN 2.1 T2V gen + optional LoRA (time-chunked VAE decode)
import os, argparse, yaml, warnings, gc
from pathlib import Path

# WAN needs ftfy for prompt cleanup
try:
    import ftfy  # noqa: F401
except Exception as e:
    raise RuntimeError("WAN pipeline requires `ftfy`.\nIn Colab:  pip -q install ftfy") from e

# Reduce CUDA fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import imageio.v2 as imageio
from tqdm import tqdm
from diffusers import DiffusionPipeline

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True

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
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try: getattr(pipe, fn)()
        except Exception: pass
    try:
        import xformers  # noqa: F401
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    except Exception:
        pass

def autocast_ctx(device, dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)

@torch.no_grad()
def decode_in_time_chunks(vae, latents: torch.Tensor, t_chunk: int = 3):
    """
    latents: (B=1, C, T, H, W) — decode in small T slices to keep VRAM low.
    Returns list of frames (HWC uint8).
    """
    # safety: keep vae on current device/dtype
    frames_out = []
    _, _, T, _, _ = latents.shape
    # WAN VAE expects the exact shape; we only slice time dimension
    for t0 in range(0, T, t_chunk):
        t1 = min(T, t0 + t_chunk)
        slab = latents[:, :, t0:t1]  # (1, C, t, H, W)
        decoded = vae.decode(slab, return_dict=False)[0]  # (1, t, H, W, 3) or similar per WAN
        # Convert to list of HWC uint8 frames
        if isinstance(decoded, torch.Tensor):
            dec = decoded.clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            dec = decoded
        # WAN returns (B, T, H, W, C); B==1
        for k in range(dec.shape[1]):
            frames_out.append(dec[0, k])
        # minimize peak memory
        del slab, decoded, dec
        torch.cuda.empty_cache()
        gc.collect()
    return frames_out

def run_one(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype, t_chunk=3):
    # Ask for LATENTS to avoid full-video decode inside the pipeline
    with torch.inference_mode(), autocast_ctx(device, dtype):
        try:
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_frames=frames,
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="latent",     # <— key change
                return_dict=True,
            )
        except torch.cuda.OutOfMemoryError:
            # Retry smaller config if we OOM during denoise
            torch.cuda.empty_cache()
            size2 = max(256, (size // 2) // 8 * 8)
            frames2 = round_wan_frames(max(9, frames - 4))
            steps2 = max(12, steps // 2)
            print(f"[OOM@denoise] Retrying with size {size}->{size2}, frames {frames}->{frames2}, steps {steps}->{steps2}")
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_frames=frames2,
                height=size2,
                width=size2,
                num_inference_steps=steps2,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="latent",
                return_dict=True,
            )
            size, frames, steps = size2, frames2, steps2

    # Chunked VAE decode on the returned latents
    latents = out.latents  # (1, C, T, H, W)
    try:
        frames_rgb = decode_in_time_chunks(pipe.vae, latents, t_chunk=t_chunk)
    except torch.cuda.OutOfMemoryError:
        # Last-ditch: cut chunk size further
        torch.cuda.empty_cache()
        gc.collect()
        print("[OOM@decode] Retrying VAE decode with smaller time chunks (t_chunk=2)")
        frames_rgb = decode_in_time_chunks(pipe.vae, latents, t_chunk=2)

    # cleanup
    del out, latents
    torch.cuda.empty_cache()
    gc.collect()
    return frames_rgb

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
    t_chunk    = int(cfg.get("sampler", {}).get("vae_t_chunk", 3))

    classes    = cfg["dataset"]["classes"]
    per_class  = int(cfg["dataset"]["per_class"])
    out_dir    = Path(cfg["dataset"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # make WAN happy re: frames
    new_frames = round_wan_frames(frames)
    if new_frames != frames:
        print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
        frames = new_frames

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,      # safe if ignored
        variant=None,
    ).to(device)

    enable_memory_savers(pipe)
    maybe_load_lora(pipe, lora_path)

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating videos …")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"
            frames_rgb = run_one(
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
                t_chunk=t_chunk,
            )
            path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(path, [to_hwc_uint8(fr) for fr in frames_rgb], fps=fps)

            # scrub between samples
            del frames_rgb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("✅ All videos generated successfully.")

if __name__ == "__main__":
    main()
