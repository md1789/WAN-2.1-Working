# generate_videos.py — WAN 2.1 T2V generation + optional LoRA (Colab-safe linalg, no ctx mgr)
import os, argparse, yaml, warnings
from pathlib import Path

# WAN's pipeline sanitizes text using ftfy
try:
    import ftfy  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "The WAN pipeline requires `ftfy`. In Colab run:\n  pip -q install ftfy"
    ) from e

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
    # WAN requires (num_frames - 1) divisible by 4; round to nearest valid
    if (n - 1) % 4 == 0:
        return n
    rounded = 4 * round((n - 1) / 4) + 1
    if rounded < 5:
        rounded = 5
    return int(rounded)


def to_hwc_uint8(frame) -> "np.ndarray":
    if isinstance(frame, torch.Tensor):
        fr = frame.detach().clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        import numpy as np
        fr = frame
        if fr.dtype != np.uint8:
            fr = fr.clip(0, 255).astype("uint8")
    return fr


def maybe_warn_lora_dir(lora_path: str):
    if not lora_path:
        return False
    if os.path.isdir(lora_path):
        return True
    print("[LoRA] WARNING: "
          f"'{lora_path}' is not a directory. Expected a folder saved via save_attn_procs(). "
          "Continuing base-only.")
    return False


def set_linalg_pref(new_pref: str):
    """
    Set CUDA linalg backend preference (e.g., 'magma' or 'cusolver') and return the previous value.
    This is a setter function, not a context manager in current PyTorch builds.
    """
    if not torch.cuda.is_available():
        return None
    prev = None
    try:
        # Query current preference (returns a torch._C._LinalgBackend or str depending on version)
        prev = torch.backends.cuda.preferred_linalg_library()
    except Exception:
        pass

    try:
        torch.backends.cuda.preferred_linalg_library(new_pref)
    except Exception as e:
        print(f"[Linalg] Could not set preferred backend to '{new_pref}': {e}")
    return prev


def autocast_ctx(device, dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    # CPU path: disable autocast
    return torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)


def run_one(pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype):
    # Prefer MAGMA to dodge sporadic cuSOLVER crashes on Colab
    prev_backend = set_linalg_pref("magma")
    try:
        with torch.no_grad(), autocast_ctx(device, dtype):
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
    except RuntimeError as e:
        msg = str(e).lower()
        if "cusolver" in msg or "cusolverdncreate" in msg:
            # One more attempt after MAGMA set (sometimes first call trips)
            with torch.no_grad(), autocast_ctx(device, dtype):
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
        raise
    finally:
        # Restore previous backend preference if we had one
        if prev_backend is not None:
            try:
                # prev_backend may be enum-like; both forms are accepted
                torch.backends.cuda.preferred_linalg_library(prev_backend)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg)

    model_name = cfg["model"].get("name", "wan-2.1-1.3b")
    base_ckpt = cfg["model"]["base_ckpt"]
    lora_path = cfg["model"].get("lora_path", "")

    seed        = int(cfg.get("sampler", {}).get("seed", 42))
    frames      = int(cfg["sampler"]["frames"])
    fps         = int(cfg["sampler"]["fps"])
    size        = int(cfg["sampler"]["size"])
    steps       = int(cfg["sampler"]["steps"])
    cfg_scale   = float(cfg["sampler"]["cfg_scale"])
    neg_prompt  = cfg["sampler"].get("negative_prompt", None)

    classes   = cfg["dataset"]["classes"]
    per_class = int(cfg["dataset"]["per_class"])
    out_dir   = Path(cfg["dataset"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # Frame rounding (quietly comply with WAN's expectation)
    new_frames = round_wan_frames(frames)
    if new_frames != frames:
        print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
        frames = new_frames

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,   # ignored by WAN if unsupported; harmless
        variant=None,
    ).to(device)

    # Keep WAN’s default scheduler (flow-matching / UniPC) — don’t override.

    # Load LoRA attention processors (directory saved via save_attn_procs)
    if maybe_warn_lora_dir(lora_path):
        try:
            pipe.transformer.load_attn_procs(lora_path)
            print(f"[LoRA] Loaded attention processors from: {lora_path}")
        except Exception as e:
            print(f"[LoRA] WARNING: Failed to load from '{lora_path}': {e}. Continuing base-only.")

    # Mild stability knobs (optional, safe)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

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

    print("✅ All videos generated successfully.")


if __name__ == "__main__":
    main()
