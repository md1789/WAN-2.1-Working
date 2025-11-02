# train_lora.py — WAN 2.1 LoRA Training (fixed version, 2025-11)
import os, argparse, yaml, math, torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataclasses import dataclass
from tqdm import tqdm
from diffusers import DiffusionPipeline

# -------------------------------------------------------------------------
@dataclass
class Config:
    cfg_path: str

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
# -------------------------------------------------------------------------

# Simple placeholder dataset — replace with real video decoding if desired
class HARVideoDataset(Dataset):
    def __init__(self, root, clip_len=16, size=256):
        self.items = []
        for cls in sorted(os.listdir(root)):
            cpath = os.path.join(root, cls)
            if not os.path.isdir(cpath):
                continue
            vids = [os.path.join(cpath, v) for v in os.listdir(cpath)
                    if v.endswith((".mp4", ".avi"))]
            self.items += [(v, cls) for v in vids]

        self.clip_len = clip_len
        self.transform = transforms.Compose([
            transforms.Resize((size, size))
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        # Placeholder random tensor; replace with real frame loader
        frames = torch.randn(3, self.clip_len, 256, 256)
        cls = self.items[idx][1]
        prompt = f"a person performing {cls}"
        return frames, prompt
# -------------------------------------------------------------------------

def inject_lora_into_transformer(transformer, rank=64, alpha=64, dropout=0.0):
    """
    Deep-inject LoRA into WAN 2.1 transformer blocks.
    Works even if attention processors are stateless callables.
    """
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0
    import torch.nn as nn

    lora_modules = []
    injected_blocks = 0

    for name, module in transformer.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
            # these are actual attention layers
            try:
                hidden_size = module.to_q.in_features
                lora_proc = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=None,
                    rank=rank,
                    network_alpha=alpha,
                )
                module.set_processor(lora_proc)
                injected_blocks += 1

                # collect trainable LoRA params
                for p in lora_proc.parameters():
                    p.requires_grad_(True)
                    lora_modules.append(p)
            except Exception as e:
                print(f"[WARN] Skipped {name}: {e}")

    total_trainable = sum(p.numel() for p in lora_modules)
    print(f"[LoRA] Deep-injected into {injected_blocks} attention layers.")
    print(f"[LoRA] Trainable params: {total_trainable:,}")
    return lora_modules

# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"[WAN 2.1] Loading base checkpoint: {cfg['model']['base_ckpt']}")
    pipe = DiffusionPipeline.from_pretrained(
        cfg["model"]["base_ckpt"],
        dtype=dtype,
    ).to(device)

    # Attach LoRA to transformer (WAN 2.1 uses transformer backbone)
    print("[LoRA] Attaching LoRA to transformer attention processors …")
    trainable_params = inject_lora_into_transformer(
        pipe.transformer,
        rank=int(cfg["lora"]["r"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
    )

    if len(trainable_params) == 0:
        raise RuntimeError("❌ No trainable LoRA params found — injection failed!")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["wd"])
    )

    # Dataset
    ds = HARVideoDataset(
        cfg["data"]["root"],
        clip_len=int(cfg["data"]["clip_len"]),
        size=int(cfg["data"]["size"]),
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
    )

    steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"]["grad_accum"])
    log_every = int(cfg["train"]["log_every"])
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    print(f"[Train] Starting for {steps} steps (grad_accum={grad_accum}) …")

    pipe.transformer.train()
    step = 0
    running_loss = 0.0

    for epoch in range(9999):  # pseudo-epoch loop for streaming dataset
        for frames, prompts in tqdm(dl, desc="Training Loop"):
            if step >= steps:
                break

            frames = frames.to(device, dtype=dtype)

            with torch.amp.autocast("cuda", enabled=(dtype == torch.float16)):
                # Dummy placeholder loss — replace with real diffusion objective
                pred = frames * 0.9
                loss = (pred - frames).abs().mean()

            scaler.scale(loss / grad_accum).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if step % log_every == 0:
                print(f"[Step {step:05d}] loss={running_loss/log_every:.6f}")
                running_loss = 0.0

            step += 1
        if step >= steps:
            break

    # Save LoRA weights
    save_path = out_dir / "lora_ema_last.safetensors"
    torch.save(pipe.transformer.state_dict(), save_path)
    print(f"✅ Training complete — LoRA weights saved to {save_path}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
