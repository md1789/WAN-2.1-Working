import os, argparse, yaml
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import WanPipeline  # DiT-based WAN pipeline (no .unet attribute)


@dataclass
class Config:
    cfg_path: str


def load_cfg(path: str):
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
            vids = [
                os.path.join(cpath, v)
                for v in os.listdir(cpath)
                if v.lower().endswith((".mp4", ".avi"))
            ]
            self.items += [(v, cls) for v in vids]
        self.clip_len = clip_len
        self.size = size
        self.transform = transforms.Compose([transforms.Resize((size, size))])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Placeholder: random tensor + prompt (replace with real video loader)
        frames = torch.randn(3, self.clip_len, self.size, self.size)  # (C, T, H, W)
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

    base_ckpt = cfg["model"].get("base_ckpt")
    if not base_ckpt:
        raise ValueError("Missing 'base_ckpt' in YAML config — check your model section.")
    print(f"Loading WAN 2.1 (1.3B) backbone from: {base_ckpt}")

    # Load WAN pipeline (DiT). NOTE: no .unet attribute on WanPipeline.
    pipe = WanPipeline.from_pretrained(base_ckpt, dtype=torch.float16).to(device)

    # ---- Dummy module to keep your training loop structure running ----
    # Replace this with real WAN LoRA adapter setup + diffusion loss later.
    dummy = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 3, 3, padding=1),
    ).to(device)

    opt = torch.optim.AdamW(
        dummy.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["wd"]),
    )

    # Data
    ds = HARVideoDataset(
        root=cfg["data"]["root"],
        clip_len=int(cfg["data"]["clip_len"]),
        size=int(cfg["data"]["size"]),
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    # Training loop (placeholder)
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    print(f"Starting dummy fine-tuning loop for {steps} steps (grad_accum={grad_accum})...")
    step = 0
    for frames, _ in tqdm(dl, total=min(len(dl), steps), ncols=100):
        if step >= steps:
            break
        # Collapse temporal dim to a single frame for the dummy op
        x = frames[:, :, 0].to(device, dtype=torch.float16)  # (B, 3, H, W)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = dummy(x)
            loss = (out - x).abs().mean()  # placeholder recon loss

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"step {step}: loss {loss.item():.4f}")
        step += 1

    # No LoRA weights to save in dummy mode
    print("Dummy training complete.")
    print(
        "Next step: swap the dummy module with WAN LoRA adapter training via Diffusers:\n"
        "  - pipe.add_adapter(...), pipe.set_adapters([...]) / load_lora_weights(...)\n"
        "  - Train adapter parameters on pipe.transformer with a denoising (diffusion) loss\n"
        "  - Then save with pipe.save_lora_weights('outputs/lora_har/lora_ema_last.safetensors')"
    )


if __name__ == "__main__":
    main()