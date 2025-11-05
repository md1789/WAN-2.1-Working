# generate_videos.py — WAN 2.1 T2V gen + optional LoRA (robust latents/frames handling, chunked decode)
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
    import numpy as np
    import torch

    # 1) torch -> numpy
    arr = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)

    # 2) If 4D, try to peel off batch/time safely without guessing channels
    #    We expect either (H,W,C,1), (1,H,W,C), or (T,H,W,C). Channels usually last.
    if arr.ndim == 4:
        # If the last dim looks like channels, keep it and drop a leading dim
        if arr.shape[-1] in (1, 3, 4):
            # prefer to drop a size-1 leading dim; otherwise take first "time" slice
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[0]  # take first frame from (T,H,W,C)
        else:
            # Last dim not channel-like: try CHW? e.g., (C,H,W,1) → move channel to end then drop last
            # Move any dim with small cardinality to the end as channels; pick the first candidate
            small_axes = [i for i, s in enumerate(arr.shape) if s in (1, 2, 3, 4)]
            if small_axes:
                arr = np.moveaxis(arr, small_axes[0], -1)
            # Now drop leading dim(s) until 3D
            while arr.ndim > 3:
                arr = arr[0]

    # 3) If 3D but CHW, fix to HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    # 4) If 2D (grayscale), replicate -> RGB
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)

    # 5) Final sanity: ensure HWC with 3 channels
    if arr.ndim != 3:
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
    H, W, C = arr.shape
    if C == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif C == 2:
        arr = np.repeat(arr[..., :1], 3, axis=2)
    elif C > 4:
        arr = arr[..., :3]

    # 6) Scale floats if in [0,1], then clamp → uint8
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8, copy=False)


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
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    try:
        import xformers  # noqa: F401
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    except Exception:
        pass


def autocast_ctx(device, dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)


def extract_latents_or_frames(output):
    """
    Return either:
      ("latents", 5D tensor)  or  ("frames", list/np/tensor already decoded)
    Handles WAN/Diffusers variants that ignore output_type='latent'.
    """
    # Preferred: latents
    if hasattr(output, "latents"):
        return "latents", output.latents
    if isinstance(output, dict) and "latents" in output:
        return "latents", output["latents"]
    # Some WAN builds stash 5D in images
    if hasattr(output, "images"):
        img = output.images
        if isinstance(img, torch.Tensor) and img.dim() == 5:
            return "latents", img
    if isinstance(output, (list, tuple)) and len(output) > 0:
        cand = output[0]
        if isinstance(cand, torch.Tensor) and cand.dim() == 5:
            return "latents", cand
        if isinstance(cand, dict) and "latents" in cand:
            return "latents", cand["latents"]

    # Fallback: decoded frames (most common when output_type is ignored)
    if hasattr(output, "frames"):
        return "frames", output.frames
    if isinstance(output, dict) and "frames" in output:
        return "frames", output["frames"]
    if hasattr(output, "images"):
        return "frames", output.images
    if isinstance(output, (list, tuple)) and len(output) > 0:
        cand = output[0]
        if isinstance(cand, (list, tuple)):
            return "frames", cand
    raise RuntimeError("Could not find latents or frames in WAN output.")


@torch.no_grad()
def decode_in_time_chunks(vae, latents: torch.Tensor, t_chunk: int = 3):
    """
    latents: (B=1, C, T, H, W) — decode in small T slices to keep VRAM low.
    Returns list of frames (HWC uint8).
    """
    frames_out = []
    _, _, T, _, _ = latents.shape
    for t0 in range(0, T, t_chunk):
        t1 = min(T, t0 + t_chunk)
        slab = latents[:, :, t0:t1]  # (1, C, t, H, W)
        decoded = vae.decode(slab, return_dict=False)[0]  # (1, t, H, W, 3)
        if isinstance(decoded, torch.Tensor):
            dec = decoded.clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            dec = decoded
        for k in range(dec.shape[1]):  # t
            frames_out.append(dec[0, k])
        del slab, decoded, dec
        torch.cuda.empty_cache()
        gc.collect()
    return frames_out

def payload_to_frame_list(payload):
    import numpy as np, torch

    # Already a list of frames?
    if isinstance(payload, list):
        out = []
        for fr in payload:
            if hasattr(fr, "numpy"):  # torch
                fr = fr.detach().cpu().numpy()
            fr = np.asarray(fr)
            # If CHW, move to HWC
            if fr.ndim == 3 and fr.shape[0] in (1,3,4) and fr.shape[-1] not in (1,3,4):
                fr = np.transpose(fr, (1,2,0))
            if fr.ndim == 2:  # gray -> RGB
                fr = np.repeat(fr[..., None], 3, axis=2)
            # scale float
            if fr.dtype.kind == "f":
                fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
            elif fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            out.append(fr)
        return out

    # Torch tensor?
    if 'torch' in str(type(payload)):
        t = payload.detach().cpu()
        if t.dim() == 5:      # (B,T,C,H,W) or (B,T,H,W,C)
            if t.size(2) in (1,3,4):     # (B,T,C,H,W)
                t = t.permute(0,1,3,4,2) # -> (B,T,H,W,C)
            t = t[0]                     # (T,H,W,C)
        elif t.dim() == 4:    # (T,C,H,W) or (T,H,W,C)
            if t.size(1) in (1,3,4):
                t = t.permute(0,2,3,1)   # -> (T,H,W,C)
        elif t.dim() == 3:    # (H,W,C) single frame
            t = t[None]                 # (1,H,W,C)
        arr = t.numpy()
    else:
        arr = np.asarray(payload)

    # NumPy paths
    if arr.ndim == 5:         # (B,T,H,W,C)
        arr = arr[0]
    if arr.ndim == 4:         # (T,H,W,C)
        frames = arr
    elif arr.ndim == 3:       # (H,W,C)
        frames = arr[None]
    else:
        raise ValueError(f"Unexpected payload shape: {arr.shape}")

    # Float->uint8
    if frames.dtype.kind == "f":
        m = frames.max()
        if m <= 1.01:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    return [frames[t] for t in range(frames.shape[0])]


def denoise_request(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype, want_latents=True):
    """
    Make one pipeline call. If want_latents is True, ask for output_type='latent'.
    Returns the raw pipeline output.
    """
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
            **({"output_type": "latent"} if want_latents else {"output_type": "np"}),
            return_dict=True,
        )


def run_one(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype, t_chunk=3):
    """
    Strategy:
      1) Try to get 'latents' (best for memory). If present, chunk-decode.
      2) If no latents but 'frames' exist, use them directly.
      3) If the call errors, retry once with output_type='np'.
    """
    try:
        out = denoise_request(
            pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype,
            want_latents=False  # <-- was True
        )
        kind, payload = extract_latents_or_frames(out)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        size2 = max(256, (size // 2) // 8 * 8)
        frames2 = round_wan_frames(max(9, frames - 4))
        steps2 = max(12, steps // 2)
        print(f"[OOM@denoise] Retrying with size {size}->{size2}, frames {frames}->{frames2}, steps {steps}->{steps2}")
        out = denoise_request(pipe, prompt, neg, frames2, size2, steps2, cfg_scale, generator, device, dtype, want_latents=True)
        kind, payload = extract_latents_or_frames(out)
        size, frames, steps = size2, frames2, steps2
    except Exception as e:
        print(f"[Warn] Latent request failed ({e}). Retrying with decoded frames.")
        out = denoise_request(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype, want_latents=False)
        kind, payload = extract_latents_or_frames(out)

    if kind == "latents":
        latents = payload  # (1, C, T, H, W)
        try:
            frames_rgb = decode_in_time_chunks(pipe.vae, latents, t_chunk=t_chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print("[OOM@decode] Retrying VAE decode with smaller time chunks (t_chunk=2)")
            frames_rgb = decode_in_time_chunks(pipe.vae, latents, t_chunk=2)
    else:
        # Already-decoded frames; normalize to HWC uint8 list
        frames_rgb = [to_hwc_uint8(fr) for fr in payload]

    frames_rgb = payload_to_frame_list(payload)
    del out
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

    new_frames = round_wan_frames(frames)
    if new_frames != frames:
        print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
        frames = new_frames

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,      # safe if ignored by WAN
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

            del frames_rgb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("✅ All videos generated successfully.")


if __name__ == "__main__":
    main()
