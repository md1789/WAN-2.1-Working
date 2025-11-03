# generate_videos.py — WAN 2.1 T2V generation with optional LoRA (folder or .pt)
import os, argparse, yaml, warnings
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import imageio.v2 as imageio
from tqdm import tqdm
from diffusers import DiffusionPipeline

# WAN's pipeline internally imports/uses ftfy during prompt cleaning
try:
    import ftfy  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "The WAN pipeline requires `ftfy`. In Colab run: `pip -q install ftfy`"
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Utilities --------------------

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_dtype(cfg) -> torch.dtype:
    bf16 = bool(cfg.get("compute", {}).get("bf16", False))
    if torch.cuda.is_available() and bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def round_frames(frames: int) -> int:
    # WAN expects (num_frames - 1) divisible by 4
    if frames < 1:
        frames = 1
    k = round((frames - 1) / 4)
    return int(1 + 4 * max(0, k))

# -------------------- LoRA (custom .pt) support --------------------
# Matches the structure saved by our train_lora.py ("lora_only.pt"):
# {
#   "lora": {
#      "<full.module.path>": {"lora_A": tensor, "lora_B": tensor, "alpha": int},
#      ...
#   },
#   "rank": int, "alpha": int, "target_patterns": [...]
# }

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear requires nn.Linear"
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / float(self.r) if self.r > 0 else 0.0
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            y = y + self.scaling * x.matmul(self.lora_A).matmul(self.lora_B.t())
        return y

def _resolve_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _ensure_lora_wrap(root: nn.Module, full_name: str, r: int, alpha: int) -> LoRALinear:
    parent, attr = _resolve_parent_and_attr(root, full_name)
    cur = getattr(parent, attr)
    if isinstance(cur, LoRALinear):
        return cur
    if not isinstance(cur, nn.Linear):
        raise TypeError(f"{full_name} is {type(cur)}; expected nn.Linear for custom LoRA")
    wrapped = LoRALinear(cur, r=r, alpha=alpha)
    setattr(parent, attr, wrapped)
    return wrapped

def load_custom_lora_pt_into_transformer(transformer: nn.Module, pt_path: str, device: str = "cuda"):
    sd: Dict = torch.load(pt_path, map_location=device)
    lora_blob: Dict = sd.get("lora", {})
    r = int(sd.get("rank", 64))
    alpha = int(sd.get("alpha", 64))
    wrapped, loaded = 0, 0
    for full_name, tensors in lora_blob.items():
        try:
            ll = _ensure_lora_wrap(transformer, full_name, r=r, alpha=alpha)
            if "lora_A" in tensors and "lora_B" in tensors:
                with torch.no_grad():
                    ll.lora_A.copy_(tensors["lora_A"].to(device))
                    ll.lora_B.copy_(tensors["lora_B"].to(device))
                loaded += 1
            wrapped += 1
        except Exception as e:
            print(f"[LoRA .pt] Skip {full_name}: {e}")
    print(f"[LoRA .pt] Wrapped {wrapped} layers; loaded tensors for {loaded} layers from {pt_path}")

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg)

    model_name = cfg["model"].get("name", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    base_ckpt  = cfg["model"].get("base_ckpt", model_name)
    lora_path  = cfg["model"].get("lora_path", "")

    frames     = round_frames(int(cfg["sampler"]["frames"]))
    fps        = int(cfg["sampler"]["fps"])
    size       = int(cfg["sampler"]["size"])
    steps      = int(cfg["sampler"]["steps"])
    cfg_scale  = float(cfg["sampler"]["cfg_scale"])
    seed       = int(cfg.get("sampler", {}).get("seed", 42))

    classes    = cfg["dataset"]["classes"]
    per_class  = int(cfg["dataset"]["per_class"])
    out_dir    = Path(cfg["dataset"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WAN 2.1 (wan-2.1-1.3b) base checkpoint and LoRA (if present) …")
    pipe = DiffusionPipeline.from_pretrained(base_ckpt).to(device)

    # LoRA (folder of attn_procs) OR custom .pt
    if lora_path:
        if os.path.isdir(lora_path):
            try:
                pipe.transformer.load_attn_procs(lora_path)
                print(f"[LoRA folder] Loaded attention processors from: {lora_path}")
            except Exception as e:
                print(f"WARNING: Failed to load attn_procs from '{lora_path}': {e}")
        elif os.path.isfile(lora_path) and lora_path.endswith(".pt"):
            try:
                load_custom_lora_pt_into_transformer(pipe.transformer, lora_path, device=device)
                print(f"[LoRA .pt] Applied custom LoRA from: {lora_path}")
            except Exception as e:
                print(f"WARNING: Failed to apply custom .pt LoRA '{lora_path}': {e}")
        else:
            print(f"WARNING: LoRA path not found or unsupported: {lora_path} — continuing base-only.")

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating videos …")
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"

            # Some WAN builds expect autocast on CUDA with fp16/bf16
            cm = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
            with torch.no_grad(), cm:
                result = pipe(
                    prompt=prompt,
                    num_frames=frames,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                )

            # Normalize outputs across diffusers versions
            frames_rgb = None
            if hasattr(result, "frames"):
                frames_rgb = result.frames
            elif hasattr(result, "videos"):
                frames_rgb = result.videos  # sometimes List[np.ndarray]
            elif isinstance(result, (list, tuple)):
                frames_rgb = result[0]
            else:
                raise RuntimeError("Unknown WAN output format; no frames/videos field found.")

            # Convert each frame to uint8 HWC for imageio
            normed = []
            for fr in frames_rgb:
                if isinstance(fr, torch.Tensor):
                    fr = fr.detach().float()
                    if fr.max() <= 1.0:  # [0,1] → [0,255]
                        fr = fr * 255.0
                    fr = fr.clamp(0, 255).to(torch.uint8).cpu().numpy()
                else:
                    # assume numpy
                    fr = fr.astype("uint8") if fr.dtype != "uint8" else fr
                # ensure HWC
                if fr.ndim == 3 and fr.shape[0] in (1,3) and fr.shape[-1] not in (1,3):
                    fr = fr.transpose(1,2,0)
                normed.append(fr)

            out_path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(out_path, normed, fps=fps)

    print("✅ All videos generated successfully.")

# nullcontext for Python 3.8+
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

if __name__ == "__main__":
    main()
