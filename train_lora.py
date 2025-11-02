# train_lora.py — WAN 2.1 LoRA training (Diffusers-compatible, version-safe)
import os, argparse, yaml
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0
import torch.nn as nn

# --------------------------- Compatibility shims ---------------------------
# Older diffusers don't export AttnProcsLayers; provide a minimal wrapper that
# exposes .named_parameters() over any module-like processors.
try:
    from diffusers.models.attention_processor import AttnProcsLayers
except ImportError:
    class AttnProcsLayers(nn.Module):
        def __init__(self, processors):
            super().__init__()
            self.processors = nn.ModuleDict({
                k: v for k, v in processors.items() if isinstance(v, nn.Module)
            })
        def forward(self, *args, **kwargs):
            raise NotImplementedError("AttnProcsLayers wrapper — no forward pass.")
        def named_parameters(self, *a, **kw):
            return self.processors.named_parameters(*a, **kw)
        def parameters(self, *a, **kw):
            return self.processors.parameters(*a, **kw)

# --------------------------- Config ---------------------------
@dataclass
class Config:
    cfg_path: str

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --------------------- Simple placeholder dataset ------------------
class HARVideoDataset(Dataset):
    """
    Placeholder dataset that returns fake frames and a text prompt.
    Replace with real video decoding as needed.
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

        # Ensure at least one dummy sample so training loop doesn't crash when empty
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

# --------------------- LoRA injection utils -------------------
def _resolve_module_by_path(root_module, dotted):
    """
    From a root nn.Module, resolve a dotted path like 'blocks.0.attn1'
    """
    cur = root_module
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def _make_lora_proc_version_safe(hidden_size, cross_dim, rank, alpha):
    """
    Construct LoRAAttnProcessor2_0 handling Diffusers signature differences.
    Tries modern args first; falls back to older ones gracefully.
    """
    try:
        # Newer diffusers (uses hidden_size + network_alpha, rank)
        return LoRAAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank,
            network_alpha=alpha,
        )
    except TypeError:
        try:
            # Some versions use 'hidden_dim' instead of 'hidden_size'
            return LoRAAttnProcessor2_0(
                hidden_dim=hidden_size,
                cross_attention_dim=cross_dim,
                rank=rank,
                network_alpha=alpha,
            )
        except TypeError:
            # Very old: often only supports (rank, cross_attention_dim)
            return LoRAAttnProcessor2_0(
                rank=rank,
                cross_attention_dim=cross_dim,
            )

def inject_lora_into_transformer(transformer, rank=64, alpha=64):
    """
    Diffusers-native LoRA injection using set_attn_processor + AttnProcsLayers.
    Works even when base attention processors are functional/stateless.

    Returns:
        lora_layers (nn.Module): AttnProcsLayers wrapper exposing trainable LoRA params.
        injected (int): number of attention modules wrapped with LoRA.
        trainable (int): count of trainable parameters across LoRA layers.
    """
    assert hasattr(transformer, "attn_processors"), \
        "Transformer has no attn_processors — unexpected WAN 2.1 structure."

    attn_procs = {}
    injected = 0

    for name, old_proc in transformer.attn_processors.items():
        # Expect names like "...attn1.processor" or "...attn2.processor"
        if not name.endswith(".processor"):
            attn_procs[name] = old_proc
            continue

        attn_module_path = name[:-len(".processor")]
        attn_module = _resolve_module_by_path(transformer, attn_module_path)
        if attn_module is None:
            attn_procs[name] = old_proc
            continue

        # Heuristic: pull the internal projection dims
        hidden_size = None
        cross_dim = getattr(attn_module, "cross_attention_dim", None)

        # Most diffusers Attention blocks have linear layers to_q/to_k/to_v
        if hasattr(attn_module, "to_q") and hasattr(attn_module.to_q, "in_features"):
            hidden_size = attn_module.to_q.in_features

        if hidden_size is None:
            # Can't infer dims — keep original processor
            attn_procs[name] = old_proc
            continue

        # Create LoRA processor in a version-safe way
        lora_proc = _make_lora_proc_version_safe(hidden_size, cross_dim, rank, alpha)
        attn_procs[name] = lora_proc
        injected += 1

    # Install new processors into the transformer
    transformer.set_attn_processor(attn_procs)

    # Wrap processors for aggregated parameter handling
    lora_layers = AttnProcsLayers(transformer.attn_processors)

    # Ensure only LoRA weights are trainable (some versions default to requires_grad=False)
    trainable = 0
    for n, p in lora_layers.named_parameters():
        # Common LoRA parameter name patterns across diffusers versions
        is_lora = (
            "lora_" in n or
            n.endswith(".alpha") or n.endswith(".scale") or
            "to_q_lora" in n or "to_k_lora" in n or "to_v_lora" in n or "to_out_lora" in n
        )
        p.requires_grad = bool(is_lora)
        if p.requires_grad:
            trainable += p.numel()

    print(f"[LoRA] Injected into {injected} attention modules.")
    print(f"[LoRA] Trainable parameters (LoRA only): {trainable:,}")
    return lora_layers, injected, trainable

# ----------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_key = str(cfg["train"].get("dtype", "fp16")).lower()
    use_fp16 = (dtype_key == "fp16")

    base_ckpt = cfg["model"].get("base_ckpt") or cfg["model"].get("ckpt")
    if not base_ckpt:
        raise ValueError("Missing base_ckpt in YAML config — check the `model` section.")

    print(f"[WAN 2.1] Loading base checkpoint: {base_ckpt}")
    # WanPipeline ignores dtype kwarg; move device/dtype after init
    pipe = DiffusionPipeline.from_pretrained(base_ckpt)
    pipe = pipe.to(device)
    if use_fp16:
        # Convert transformer to fp16 where supported (safe no-op on layers that don't support it)
        try:
            pipe.transformer.to(dtype=torch.float16)
        except Exception:
            pass

    # ---------------- LoRA injection ----------------
    print("[LoRA] Attaching LoRA to transformer attention processors …")
    rank = int(cfg["lora"]["r"])
    alpha = int(cfg["lora"]["alpha"])
    lora_layers, injected, trainable = inject_lora_into_transformer(
        pipe.transformer, rank=rank, alpha=alpha
    )
    if injected == 0 or trainable == 0:
        raise RuntimeError("❌ No trainable LoRA params found — injection failed!")

    # --------------- Optimizer over LoRA only ---------------------
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["wd"])
    opt = torch.optim.AdamW(
        (p for p in lora_layers.parameters() if p.requires_grad),
        lr=lr, weight_decay=wd
    )

    # ---------------------- Data -------------------------------
    ds = HARVideoDataset(
        cfg["data"]["root"],
        clip_len=int(cfg["data"]["clip_len"]),
        size=int(cfg["data"]["size"]),
    )
    # Handle empty dir gracefully (dataset stub ensures >=1)
    batch_size = max(1, int(cfg["train"]["batch_size"]))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # ------------------- Training loop (placeholder) -------------
    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"[Train] Starting for {steps} steps (grad_accum={grad_accum}) …")
    pipe.transformer.train()
    step = 0
    opt.zero_grad(set_to_none=True)

    # NOTE: placeholder loss just to exercise LoRA params; replace with real diffusion loss.
    for frames, _ in tqdm(dl):
        if step >= steps:
            break
        with torch.amp.autocast("cuda", enabled=use_fp16):
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for p in lora_layers.parameters():
                if p.requires_grad:
                    # ultra-tiny term that touches LoRA params to form a grad graph
                    loss = loss + (p.float() * 0.0).sum()
            loss = loss + 1e-6

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"step {step}: placeholder loss {float(loss):.6f}")

        step += 1
        if step >= steps:
            break

    # ----------------- Save LoRA weights -------------------------
    save_path = out_dir / "lora_ema_last"
    pipe.transformer.save_attn_procs(save_path)
    print(f"✅ Training complete — LoRA adapters saved to {save_path}")

if __name__ == "__main__":
    main()
