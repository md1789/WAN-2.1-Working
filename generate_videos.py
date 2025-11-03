# generate_videos.py — WAN 2.1 T2V generation + optional LoRA + scheduler switch
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
from diffusers.schedulers import (
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
)

warnings.filterwarnings("ignore", category=UserWarning)


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_dtype(cfg):
    bf16 = bool(cfg.get("compute", {}).get("bf16", False))
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_scheduler(pipe, name: str):
    name = (name or "euler").lower()
    try:
        if name in ["euler", "euler_discrete"]:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif name in ["dpmpp_2m", "dpmpp", "dpm++", "dpmsolver++"]:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++"
            )
        elif name in ["ddim"]:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif name in ["unipc", "unipc_multistep"]:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            print(f"[Scheduler] Unknown '{name}', defaulting to Euler.")
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        print(f"[Scheduler] Failed to set '{name}' ({e}); falling back to Euler.")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    print(f"[Scheduler] Using: {pipe.scheduler.__class__.__name__}")


def round_wan_frames(n: int) -> int:
    # WAN requires (num_frames - 1) divisible by 4; round to nearest valid
    if (n - 1) % 4 == 0:
        return n
    rounded = 4 * round((n - 1) / 4) + 1
    if rounded < 5:
        rounded = 5  # minimal sane sequence
    return int(rounded)


def to_hwc_uint8(frame) -> "np.ndarray":
    if isinstance(frame, torch.Tensor):
        fr = frame.detach().clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        # assume numpy
        fr = frame
        # trust pipeline outputs are already uint8-range; enforce type
        import numpy as np

        if fr.dtype != np.uint8:
            fr = fr.clip(0, 255).astype("uint8")
    return fr


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

    seed = int(cfg.get("sampler", {}).get("seed", 42))
    frames = int(cfg["sampler"]["frames"])
    fps = int(cfg["sampler"]["fps"])
    size = int(cfg["sampler"]["size"])
    steps = int(cfg["sampler"]["steps"])
    cfg_scale = float(cfg["sampler"]["cfg_scale"])
    neg_prompt = cfg["sampler"].get("negative_prompt", None)
    scheduler_name = cfg.get("sampler", {}).get("scheduler", "euler")

    classes = cfg["dataset"]["classes"]
    per_class = int(cfg["dataset"]["per_class"])
    out_dir = Path(cfg["dataset"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frame rounding (quietly comply with WAN's expectation)
    new_frames = round_wan_frames(frames)
    if new_frames != frames:
        print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
        frames = new_frames

    print(f"Loading WAN 2.1 ({model_name}) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(
        base_ckpt,
        dtype=dtype,   # modern diffusers prefers `dtype`
        variant=None,
    ).to(device)

    # Swap scheduler if requested
    set_scheduler(pipe, scheduler_name)

    # Load LoRA (directory saved via `save_attn_procs`)
    if lora_path:
        if os.path.isdir(lora_path):
            try:
                pipe.transformer.load_attn_procs(lora_path)
                print(f"[LoRA] Loaded attention processors from: {lora_path}")
            except Exception as e:
                print(f"[LoRA] WARNING: Failed to load from '{lora_path}': {e}. Continuing base-only.")
        else:
            print(f"[LoRA] WARNING: '{lora_path}' is not a directory. Expected a folder saved via save_attn_procs(). Continuing base-only.")

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating videos …")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"

            ctx = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if device == "cuda"
                else torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)
            )

            with torch.no_grad(), ctx:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_frames=frames,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                )

            # Diffusers WAN returns a list of frames
            frames_rgb = [to_hwc_uint8(fr) for fr in result.frames]

            path = cdir / f"{cls}_{i:02d}.mp4"
            # imageio uses installed imageio-ffmpeg in Colab by default
            imageio.mimwrite(path, frames_rgb, fps=fps)

    print("✅ All videos generated successfully.")


if __name__ == "__main__":
    main()
