# train_lora.py — WAN 2.1 LoRA training
import os, argparse, math, time, yaml
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from diffusers import DiffusionPipeline
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

@dataclass
class Config:
    cfg_path: str

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------------------------------
# Simple placeholder dataset — replace with real frame decoding if desired
class HARVideoDataset(Dataset):
    def __init__(self, root, clip_len=16, size=256):
        self.items = []
        for cls in sorted(os.listdir(root)):
            cpath = os.path.join(root, cls)
            if not os.path.isdir(cpath): 
                continue
            vids = [os.path.join(cpath, v) for v in os.listdir(cpath) if v.endswith((".mp4", ".avi"))]
            self.items += [(v, cls) for v in vids]
        self.clip_len = clip_len
        self.transform = transforms.Compose([transforms.Resize((size, size))])

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        # Placeholder: random tensor + prompt (replace with real video loader)
        frames = torch.randn(3, self.clip_len, 256, 256)
        cls = self.items[idx][1]
        prompt = f"a person performing {cls}"
        return frames, prompt
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading WAN 2.1 1.3 G backbone...")
    pipe = DiffusionPipeline.from_pretrained(
        cfg["model"]["ckpt"],
        torch_dtype=torch.float16,
    ).to(device)

    # Freeze base model
    for p in pipe.unet.parameters():
        p.requires_grad = False

    # Inject LoRA
    print("Injecting LoRA adapters...")
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=float(cfg["lora"]["dropout"]),
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    # Optimizer
    opt = torch.optim.AdamW(
        pipe.unet.parameters(), 
        lr=float(cfg["train"]["lr"]), 
        weight_decay=float(cfg["train"]["wd"])
    )

    # Data
    ds = HARVideoDataset(
        cfg["data"]["root"], 
        clip_len=int(cfg["data"]["clip_len"]), 
        size=int(cfg["data"]["size"])
    )
    dl = DataLoader(
        ds, 
        batch_size=int(cfg["train"]["batch_size"]), 
        shuffle=True, 
        num_workers=2
    )

    # Training loop (lightweight placeholder — add diffusion loss for real training)
    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])

    scaler = torch.cuda.amp.GradScaler()
    print(f"Starting LoRA fine-tuning for {steps} steps...")
    for step, batch in enumerate(tqdm(dl)):
        if step >= steps:
            break
        frames, prompts = batch
        frames = frames.to(device, dtype=torch.float16)

        # Dummy reconstruction loss (replace with diffusion loss later)
        with torch.cuda.amp.autocast():
            out = frames * 0.9  # placeholder transform
            loss = (out - frames).abs().mean()

        scaler.scale(loss).backward()
        if (step + 1) % cfg["train"]["grad_accum"] == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        if step % log_every == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    # Save LoRA weights
    save_path = out_dir / "lora_ema_last.safetensors"
    pipe.unet.save_attn_procs(save_path)
    print(f"✅ Training complete — LoRA weights saved to {save_path}")

if __name__ == "__main__":
    main()
