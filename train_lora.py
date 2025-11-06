# train_lora.py — WAN 2.1 LoRA training (module-agnostic Linear patch)
# Works when WanPipeline doesn't expose diffusers attn_processors.
import os, argparse, yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import DiffusionPipeline


# --------------------------- Config ---------------------------

@dataclass
class Config:
    cfg_path: str

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------- Simple placeholder ds ------------------

class HARVideoDataset(Dataset):
    """
    Placeholder dataset:
      returns fake frames and a text prompt; replace with real video frame decoding.
    """
    def __init__(self, root, clip_len=16, size=256):
        self.items = []
        if os.path.isdir(root):
            for cls in sorted(os.listdir(root)):
                cpath = os.path.join(root, cls)
                if not os.path.isdir(cpath):
                    continue
                vids = [os.path.join(cpath, v)
                        for v in os.listdir(cpath)
                        if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
                self.items += [(v, cls) for v in vids]
        # If no data found, keep one dummy sample so the loop runs
        if len(self.items) == 0:
            self.items = [("dummy.mp4", "action")]

        self.clip_len = int(clip_len)
        self.size = int(size)
        self.transform = transforms.Compose([transforms.Resize((size, size))])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        frames = torch.randn(3, self.clip_len, self.size, self.size)
        cls = self.items[idx][1]
        prompt = f"a person performing {cls}"
        return frames, prompt


# ------------------------ LoRA modules ------------------------

class LoRALinear(nn.Module):
    """
    Drop-in Linear wrapper with LoRA adapters:
      y = base(x) + scale * x @ A @ B^T
    Base linear is frozen; only A/B train.
    """
    def __init__(self, base_linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear requires nn.Linear"
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias is not None

        # keep original as a submodule (frozen)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / float(self.r) if self.r > 0 else 0.0

        if self.r > 0:
            # A: in_features x r, B: out_features x r (we'll do x @ A @ B^T)
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            # init following LoRA paper small init
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            # safety: degenerate no-op
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # x: [*, in]; A: [in, r]; B: [out, r]
            # delta = x @ A @ B^T
            delta = x.matmul(self.lora_A).matmul(self.lora_B.t())
            y = y + self.scaling * delta
        return y


# ---------------------- Model surgery utils -------------------

def _resolve_parent_and_attr(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """
    Given 'blocks.0.attn1.to_q', return (parent_module, 'to_q').
    """
    parts = dotted.split(".")
    attr = parts[-1]
    parent_path = ".".join(parts[:-1])
    parent = root
    if parent_path:
        for p in parts[:-1]:
            parent = getattr(parent, p)
    return parent, attr

def _name_matches_any(name: str, patterns: List[str]) -> bool:
    name_low = name.lower()
    return any(pat.lower() in name_low for pat in patterns)

def inject_lora_into_linears(
    root: nn.Module,
    target_patterns: List[str],
    r: int,
    alpha: int
) -> Tuple[int, int]:
    """
    Walk model; whenever a module name matches any pattern and is nn.Linear,
    replace with LoRALinear. Returns (#replaced, #trainable_params).
    """
    replaced = 0
    trainable = 0

    # Collect candidates first to avoid modifying while iterating
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Linear) and _name_matches_any(name, target_patterns):
            candidates.append((name, mod))

    # Perform replacements
    for full_name, linear in candidates:
        parent, attr = _resolve_parent_and_attr(root, full_name)
        new_mod = LoRALinear(linear, r=r, alpha=alpha)
        setattr(parent, attr, new_mod)
        replaced += 1
        for p in new_mod.parameters():
            if p.requires_grad:
                trainable += p.numel()

    return replaced, trainable


def collect_lora_state_dict(root: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Gather only LoRA adapter tensors for saving. Returns dict:
      { 'module_full_name': {'lora_A': ..., 'lora_B': ..., 'alpha': tensor([...])}, ... }
    """
    out = {}
    for name, mod in root.named_modules():
        if isinstance(mod, LoRALinear) and mod.r > 0:
            out[name] = {
                "lora_A": mod.lora_A.detach().cpu(),
                "lora_B": mod.lora_B.detach().cpu(),
                "alpha": torch.tensor(mod.alpha, dtype=torch.int32)
            }
    return out


# ----------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = str(cfg["train"].get("dtype", "fp16")).lower() == "fp16"

    base_ckpt = cfg["model"].get("base_ckpt") or cfg["model"].get("ckpt")
    if not base_ckpt:
        raise ValueError("Missing base_ckpt in YAML config — check the `model` section.")

    print(f"[WAN 2.1] Loading base checkpoint: {base_ckpt}")
    # Don’t pass dtype kwarg (WanPipeline ignores/complains); move to device afterward
    pipe = DiffusionPipeline.from_pretrained(base_ckpt)
    pipe.to(device)

    # ---------------- LoRA injection (module-agnostic) ----------------
    default_patterns = [
        # common diffusers names
        "to_q", "to_k", "to_v", "to_out",
        # common HF/transformer names
        "q_proj", "k_proj", "v_proj", "o_proj",
        # just in case
        "attn", "attention",
    ]
    target_patterns = cfg.get("lora", {}).get("target_patterns", default_patterns)
    r = int(cfg["lora"].get("r", 64))
    alpha = int(cfg["lora"].get("alpha", 64))

    print(f"[LoRA] Target patterns: {target_patterns}")
    print(f"[LoRA] Injecting LoRA (r={r}, alpha={alpha}) into matching Linear layers …")

    # Prefer the video transformer if present; else try whole pipe
    target_root = getattr(pipe, "transformer", pipe)
    replaced, trainable = inject_lora_into_linears(
        target_root, target_patterns=target_patterns, r=r, alpha=alpha
    )
    print(f"[LoRA] Wrapped {replaced} Linear modules with LoRA.")
    print(f"[LoRA] Trainable parameters (LoRA only): {trainable:,}")

    if replaced == 0 or trainable == 0:
        raise RuntimeError("❌ No trainable LoRA params found — injection failed (no matching Linear layers).")

    # --------------- Optimizer over LoRA only ---------------------
    lr = float(cfg["train"].get("lr", 1e-4))
    wd = float(cfg["train"].get("wd", 0.0))
    lora_params = [p for m in target_root.modules() for p in getattr(m, "parameters", lambda: [])() if isinstance(m, LoRALinear) and p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=wd)

    # ---------------------- Data -------------------------------
    ds = HARVideoDataset(
        cfg["data"]["root"],
        clip_len=int(cfg["data"].get("clip_len", 16)),
        size=int(cfg["data"].get("size", 256)),
    )
    batch_size = max(1, int(cfg["train"].get("batch_size", 1)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # ------------------- Training loop (placeholder) -------------
    out_dir = Path(cfg["train"].get("out_dir", "outputs/lora")); out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["train"].get("max_steps", 200))
    log_every = int(cfg["train"].get("log_every", 10))
    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))

    scaler = torch.amp.GradScaler("cuda", enabled=(use_fp16 and (device == "cuda")))
    dtype_ctx = torch.amp.autocast("cuda", enabled=(use_fp16 and (device == "cuda")))

    print(f"[Train] Starting for {steps} steps (grad_accum={grad_accum}) …")
    target_root.train()
    step = 0
    opt.zero_grad(set_to_none=True)

    for frames, _ in tqdm(dl):
        if step >= steps:
            break

        # ✨ Placeholder objective that touches LoRA params (replace with real diffusion loss)
        with dtype_ctx:
            loss = torch.zeros((), device=device, dtype=torch.float32)
            # L2 penalty on LoRA weights just to produce gradients
            for m in target_root.modules():
                if isinstance(m, LoRALinear) and m.r > 0:
                    loss = loss + (m.lora_A.float().pow(2).mean() + m.lora_B.float().pow(2).mean()) * 1e-6
            loss = loss + 1e-6  # ensure non-zero

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"step {step}: placeholder loss {float(loss):.8f}")

        step += 1

    # ----------------- Save LoRA adapters -------------------------
    lora_sd = collect_lora_state_dict(target_root)
    save_path = out_dir / "lora_only.pt"
    torch.save({"lora": lora_sd, "rank": r, "alpha": alpha, "target_patterns": target_patterns}, save_path)
    print(f"Training complete — LoRA adapters saved to {save_path}")

if __name__ == "__main__":
    main()
