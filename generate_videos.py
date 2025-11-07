from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import yaml
import torch
import numpy as np
from typing import List

try:
    from diffusers import AutoPipelineForText2Video, DPMSolverMultistepScheduler
except Exception as e:
    print("[FATAL] diffusers import failed:", e, file=sys.stderr)
    raise

try:
    import imageio.v2 as imageio
except Exception as e:
    print("[FATAL] imageio import failed:", e, file=sys.stderr)
    raise


def read_out_dir_from_yaml(cfg_path: Path | str) -> Path | None:
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Expected layout: dataset: { out_dir: "outputs/samples" }
        out_dir = None
        if isinstance(data, dict):
            ds = data.get("dataset") or data.get("data") or {}
            out_dir = ds.get("out_dir") or ds.get("output_dir")
        if out_dir:
            return Path(out_dir)
    return None


def make_pipe(model_id: str,
              use_bf16: bool = False,
              disable_xformers: bool = True) -> AutoPipelineForText2Video:
    has_cuda = torch.cuda.is_available()
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 requested but not supported on this GPU; falling back to fp16.")
        use_bf16 = False

    dtype = torch.bfloat16 if (use_bf16 and has_cuda) else (torch.float16 if has_cuda else torch.float32)

    print(f"[info] Creating pipeline: dtype={dtype}, device={'cuda' if has_cuda else 'cpu'}")
    pipe = AutoPipelineForText2Video.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype in (torch.float16, torch.bfloat16) else None,
        use_safetensors=True,
    )

    # Stable scheduler (Karras)
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        print("[info] Using DPMSolverMultistepScheduler (Karras)")
    except Exception as e:
        print("[warn] Could not swap scheduler:", e)

    # Optional: disabling memory-efficient attention if it causes instability
    if disable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention(False)
            print("[info] Disabled xFormers memory-efficient attention")
        except Exception:
            pass

    device = torch.device("cuda" if has_cuda else "cpu")
    pipe.to(device)
    return pipe


def sanitize_frames_tensor(frames_t: torch.Tensor) -> np.ndarray:
    """Map to [0,1], remove NaNs/Infs, clamp, return uint8 numpy (T,H,W,C)."""
    if frames_t.ndim == 3:
        # (T, H, W) → (T, H, W, 1)
        frames_t = frames_t.unsqueeze(-1)

    # Some pipelines output in [-1, 1]; map if needed.
    try:
        if torch.nanmin(frames_t) < 0:
            frames_t = (frames_t.clamp(-1, 1) + 1.0) / 2.0
    except Exception:
        # If nanmin fails due to all-NaN, fall back to zeroing below
        pass

    frames_t = torch.nan_to_num(frames_t, nan=0.0, posinf=1.0, neginf=0.0)
    frames_t = frames_t.clamp(0.0, 1.0)
    frames_u8 = (frames_t * 255).round().to(torch.uint8).cpu().numpy()
    return frames_u8


def pil_list_to_uint8(frames: List) -> np.ndarray:
    # frames: list of PIL Images
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in frames], axis=0)
    return arr


def save_mp4(frames_u8: np.ndarray, out_path: Path, fps: int = 24) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # imageio handles (T,H,W,C) uint8
    imageio.mimsave(out_path.as_posix(), frames_u8, fps=fps, quality=8)  # quality 0-10 (for ffmpeg backend)
    print(f"[ok] Wrote {out_path}")


def generate_action(pipe,
                    action: str,
                    prompt: str,
                    negative_prompt: str,
                    steps: int,
                    guidance: float,
                    height: int,
                    width: int,
                    num_frames: int,
                    fps: int,
                    seed: int,
                    out_root: Path) -> None:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        output_type="pil",  # force decode
    )

    # Robustly extract frames
    frames = None
    if hasattr(out, "frames") and out.frames:
        frames = out.frames[0]
    elif hasattr(out, "images") and out.images:
        frames = out.images
    elif isinstance(out, (list, tuple)) and out and hasattr(out[0], "images"):
        frames = out[0].images

    if frames is None:
        # As a last resort, try to sanitize any latent-like tensor
        if hasattr(out, "videos") and isinstance(out.videos, torch.Tensor):
            frames_u8 = sanitize_frames_tensor(out.videos)
        else:
            raise RuntimeError("Could not find decoded frames in pipeline output.")
    else:
        frames_u8 = pil_list_to_uint8(frames)

    # Save
    out_dir = out_root / "framepack" / action
    out_path = out_dir / f"{action}_00.mp4"
    save_mp4(frames_u8, out_path, fps=fps)


def main():
    p = argparse.ArgumentParser(description="Stable WAN 2.1 video generator")
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--cfg", dest="cfg", default="configs/infer_har.yaml")
    p.add_argument("--out_dir", default=None, help="Override output directory; else read from YAML")
    p.add_argument("--actions", nargs="*", default=["walking", "running", "jumping", "waving"]) 
    p.add_argument("--prompt_template", default="A cinematic {action} shot of a single person, realistic lighting, shallow depth of field")
    p.add_argument("--negative_prompt", default="low quality, artifacts, distorted limbs, blurry, watermark")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=48)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="Try bf16 (falls back if unsupported)")
    p.add_argument("--keep_xformers", action="store_true", help="Keep xFormers attention enabled")
    args = p.parse_args()

    out_root = Path(args.out_dir) if args.out_dir else (read_out_dir_from_yaml(args.cfg) or Path("outputs/samples"))
    print("[info] Output root:", out_root)

    # Build pipeline
    pipe = make_pipe(args.model_id, use_bf16=args.bf16, disable_xformers=(not args.keep_xformers))

    # Generate each action
    for action in args.actions:
        prompt = args.prompt_template.format(action=action)
        print(f"[gen] {action}: steps={args.steps}, guidance={args.guidance}, size={args.width}x{args.height}, frames={args.num_frames}, fps={args.fps}")
        generate_action(
            pipe=pipe,
            action=action,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
            out_root=out_root,
        )

    print("[done] All videos generated.")


if __name__ == "__main__":
    main()
