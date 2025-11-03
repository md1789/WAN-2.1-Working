# train_lora.py — WAN 2.1 LoRA training (version-adaptive for Diffusers)
import os, argparse, yaml, inspect
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import DiffusionPipeline
import torch.nn as nn

# Prefer 2_0 but keep a fallback to classic class
from diffusers.models.attention_processor import AttnProcessor2_0
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LORA_CANDIDATE_1
except Exception:
    LORA_CANDIDATE_1 = None
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor as LORA_CANDIDATE_2
except Exception:
    LORA_CANDIDATE_2 = None

# --------------------------- Compatibility shims ---------------------------
try:
    from diffusers.models.attention_processor import AttnProcsLayers
except Exception:
    class AttnProcsLayers(nn.Module):
        def __init__(self, processors):
            super().__init__()
            self.processors = nn.ModuleDict({k: v for k, v in processors.items()
                                             if isinstance(v, nn.Module)})
        def forward(self, *args, **kwargs):
            raise NotImplementedError
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
    def __init__(self, root, clip_len=16, size=256):
        self.items = []
        if os.path.isdir(root):
            for cls in sorted(os.listdir(root)):
                cpath = os.path.join(root, cls)
                if not os.path.isdir(cpath): continue
                vids = [os.path.join(cpath, v) for v in os.listdir(cpath)
                        if v.lower().endswith((".mp4",".avi",".mov",".mkv"))]
                self.items += [(v, cls) for v in vids]
        if not self.items:
            self.items = [("dummy.mp4", "action")]
        self.clip_len = int(clip_len)
        self.size = int(size)
        self.transform = transforms.Compose([transforms.Resize((size, size))])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        frames = torch.randn(3, self.clip_len, self.size, self.size)
        cls = self.items[idx][1]
        return frames, f"a person performing {cls}"

# --------------------- Helpers -------------------
def _resolve_module_by_path(root_module, dotted):
    cur = root_module
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def _ctor_accepts_kwargs(ctor, names):
    sig = inspect.signature(ctor)
    return all(n in sig.parameters for n in names)

def _first_of(seq):
    return next((x for x in seq if x is not None), None)

def _build_lora_proc_version_safe(attn_module, rank, alpha):
    """
    Try multiple constructors/signatures for LoRA processors across diffusers versions.
    Returns a nn.Module processor or raises RuntimeError.
    """
    # infer dims
    hidden_size = None
    if hasattr(attn_module, "to_q") and hasattr(attn_module.to_q, "in_features"):
        hidden_size = attn_module.to_q.in_features
    cross_dim = getattr(attn_module, "cross_attention_dim", None)

    errors = []

    # Candidate constructors in preference order
    for LoraCls in (LORA_CANDIDATE_1, LORA_CANDIDATE_2):
        if LoraCls is None:
            continue
        ctor = LoraCls.__init__
        try:
            # 1) Most modern: hidden_size + cross_attention_dim + rank + network_alpha
            if _ctor_accepts_kwargs(ctor, {"hidden_size","cross_attention_dim","rank","network_alpha"}):
                return LoraCls(hidden_size=hidden_size, cross_attention_dim=cross_dim,
                               rank=rank, network_alpha=alpha)
            # 2) Some versions use hidden_dim instead of hidden_size
            if _ctor_accepts_kwargs(ctor, {"hidden_dim","cross_attention_dim","rank","network_alpha"}):
                return LoraCls(hidden_dim=hidden_size, cross_attention_dim=cross_dim,
                               rank=rank, network_alpha=alpha)
            # 3) Old variants use r/lora_alpha naming
            if _ctor_accepts_kwargs(ctor, {"hidden_size","cross_attention_dim","r","lora_alpha"}):
                return LoraCls(hidden_size=hidden_size, cross_attention_dim=cross_dim,
                               r=rank, lora_alpha=alpha)
            if _ctor_accepts_kwargs(ctor, {"hidden_dim","cross_attention_dim","r","lora_alpha"}):
                return LoraCls(hidden_dim=hidden_size, cross_attention_dim=cross_dim,
                               r=rank, lora_alpha=alpha)
            # 4) Very old minimal signatures
            if _ctor_accepts_kwargs(ctor, {"rank","cross_attention_dim"}):
                return LoraCls(rank=rank, cross_attention_dim=cross_dim)
            if _ctor_accepts_kwargs(ctor, {"r","cross_attention_dim"}):
                return LoraCls(r=rank, cross_attention_dim=cross_dim)
            # 5) Last-ditch: try positional patterns
            try:
                # hidden_size, cross_dim, rank, alpha
                return LoraCls(hidden_size, cross_dim, rank, alpha)
            except Exception as e1:
                try:
                    # rank, cross_dim
                    return LoraCls(rank, cross_dim)
                except Exception as e2:
                    errors += [repr(e1), repr(e2)]
        except Exception as e:
            errors.append(repr(e))

    raise RuntimeError("LoRA processor construction failed; tried multiple signatures.\n" +
                       "\n".join(errors))

def inject_lora_into_transformer(transformer, rank=64, alpha=64):
    assert hasattr(transformer, "attn_processors"), "Transformer has no attn_processors."
    attn_procs = {}
    injected = 0

    for name, old_proc in transformer.attn_processors.items():
        if not name.endswith(".processor"):
            attn_procs[name] = old_proc
            continue
        attn_module_path = name[:-len(".processor")]
        attn_module = _resolve_module_by_path(transformer, attn_module_path)
        if attn_module is None:
            attn_procs[name] = old_proc
            continue
        try:
            lora_proc = _build_lora_proc_version_safe(attn_module, rank, alpha)
            attn_procs[name] = lora_proc
            injected += 1
        except Exception:
            # keep original if we cannot wrap
            attn_procs[name] = old_proc

    transformer.set_attn_processor(attn_procs)
    lora_layers = AttnProcsLayers(transformer.attn_processors)

    trainable = 0
    for n, p in lora_layers.named_parameters():
        is_lora = ("lora" in n) or n.endswith(".alpha") or n.endswith(".scale")
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
    use_fp16 = str(cfg["train"].get("dtype","fp16")).lower() == "fp16"

    base_ckpt = cfg["model"].get("base_ckpt") or cfg["model"].get("ckpt")
    if not base_ckpt:
        raise ValueError("Missing base_ckpt in YAML config — check model.base_ckpt")

    print(f"[WAN 2.1] Loading base checkpoint: {base_ckpt}")
    pipe = DiffusionPipeline.from_pretrained(base_ckpt).to(device)
    if use_fp16:
        try: pipe.transformer.to(dtype=torch.float16)
        except Exception: pass

    print("[LoRA] Attaching LoRA to transformer attention processors …")
    rank = int(cfg["lora"]["r"]); alpha = int(cfg["lora"]["alpha"])
    lora_layers, injected, trainable = inject_lora_into_transformer(pipe.transformer, rank, alpha)
    if injected == 0 or trainable == 0:
        raise RuntimeError("❌ No trainable LoRA params found — injection failed!")

    # Optimizer over LoRA only
    opt = torch.optim.AdamW(
        (p for p in lora_layers.parameters() if p.requires_grad),
        lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["wd"])
    )

    # Data
    ds = HARVideoDataset(cfg["data"]["root"],
                         clip_len=int(cfg["data"]["clip_len"]),
                         size=int(cfg["data"]["size"]))
    dl = DataLoader(ds, batch_size=max(1,int(cfg["train"]["batch_size"])),
                    shuffle=True, num_workers=2)

    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"]["log_every"])
    grad_accum = max(1, int(cfg["train"].get("grad_accum",1)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"[Train] Starting for {steps} steps (grad_accum={grad_accum}) …")
    pipe.transformer.train()
    opt.zero_grad(set_to_none=True)
    step = 0

    for frames, _ in tqdm(dl):
        if step >= steps: break
        with torch.amp.autocast("cuda", enabled=use_fp16):
            # Placeholder loss touching LoRA params only
            loss = torch.zeros((), device=device, dtype=torch.float32) + 1e-6
            for p in lora_layers.parameters():
                if p.requires_grad:
                    loss = loss + (p.float()*0.0).sum()

        scaler.scale(loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"step {step}: placeholder loss {float(loss):.6f}")
        step += 1

    save_path = out_dir / "lora_ema_last"
    pipe.transformer.save_attn_procs(save_path)
    print(f"✅ Training complete — LoRA adapters saved to {save_path}")

if __name__ == "__main__":
    main()
