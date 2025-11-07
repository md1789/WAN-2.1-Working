# generate_videos.py — WAN 2.1 T2V gen + optional LoRA + FramePack backend
import os, argparse, yaml, warnings, gc
from pathlib import Path

# WAN needs ftfy for prompt cleanup
try:
    import ftfy  # noqa: F401
except Exception as e:
    raise RuntimeError("WAN pipeline requires `ftfy`.\nIn Colab:  pip -q install ftfy") from e

# Reduce CUDA fragmentation / noisy tokenizer threading
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm

from diffusers import DiffusionPipeline
from diffusers import (
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)

# --- SigLIP image processor import shim (transformers version tolerant) ---
try:
    # Transformers ≥ ~4.41
    from transformers import SiglipImageProcessor
except Exception:
    try:
        # Works across many versions
        from transformers import AutoImageProcessor as SiglipImageProcessor
    except Exception:
        # Very new unified API fallback
        from transformers import AutoProcessor as SiglipImageProcessor
# -------------------------------------------------------------------------

# --- SigLIP vision encoder import shim (transformers version tolerant) ---
try:
    # Preferred
    from transformers import SiglipVisionModel
except Exception:
    try:
        # Generic vision model fallback
        from transformers import AutoModel as SiglipVisionModel
    except Exception:
        # Last resort (not ideal, but unblocks envs)
        from transformers import CLIPVisionModel as SiglipVisionModel
# -------------------------------------------------------------------------

# --- Optional T2I for priming FramePack with real keyframes ---
def make_keyframes_from_prompt(prompt, size, dtype, device, seed1=1234, seed2=1337, model_id="black-forest-labs/FLUX.1-dev"):
    from diffusers import DiffusionPipeline
    import numpy as _np
    t2i = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    h = w = size
    g1 = torch.Generator(device=device).manual_seed(seed1)
    g2 = torch.Generator(device=device).manual_seed(seed2)
    img1 = t2i(prompt=prompt, height=h, width=w, num_inference_steps=28, guidance_scale=4.0, generator=g1).images[0]
    # subtle motion/pose hint
    img2 = t2i(prompt=prompt + ", different pose, motion blur, dynamic lighting",
               height=h, width=w, num_inference_steps=28, guidance_scale=4.5, generator=g2).images[0]
    return _np.asarray(img1), _np.asarray(img2)

def load_keyframes_from_dir(init_dir, size):
    # returns [first, last] as np.uint8 HWC
    import glob, cv2, numpy as _np
    paths = sorted([p for p in glob.glob(os.path.join(init_dir, "*")) if os.path.splitext(p)[1].lower() in [".png",".jpg",".jpeg",".webp"]])
    if not paths:
        raise ValueError(f"No keyframe images found in {init_dir}")
    def _read(path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None: raise ValueError(f"Failed to read {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        return im
    first = _read(paths[0])
    last  = _read(paths[-1]) if len(paths) > 1 else None
    return [first, last] if last is not None else [first]


warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True


# ------------------------- utils -------------------------

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_dtype(cfg):
    bf16 = bool(cfg.get("compute", {}).get("bf16", False))
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def round_wan_frames(n: int) -> int:
    # WAN requires (num_frames - 1) % 4 == 0
    if (n - 1) % 4 == 0:
        return n
    r = 4 * round((n - 1) / 4) + 1
    return int(max(5, r))


def to_hwc_uint8(frame) -> "np.ndarray":
    # Accepts tensor/np/list; returns HWC uint8
    import numpy as _np
    import torch as _torch

    arr = frame.detach().cpu().numpy() if isinstance(frame, _torch.Tensor) else _np.asarray(frame)

    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[0]
        else:
            small_axes = [i for i, s in enumerate(arr.shape) if s in (1, 2, 3, 4)]
            if small_axes:
                arr = _np.moveaxis(arr, small_axes[0], -1)
            while arr.ndim > 3:
                arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = _np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    if arr.ndim == 2:
        arr = _np.repeat(arr[..., None], 3, axis=2)

    if arr.ndim != 3:
        arr = _np.zeros((8, 8, 3), dtype=_np.uint8)

    H, W, C = arr.shape
    if C == 1:
        arr = _np.repeat(arr, 3, axis=2)
    elif C == 2:
        arr = _np.repeat(arr[..., :1], 3, axis=2)
    elif C > 4:
        arr = arr[..., :3]

    if _np.issubdtype(arr.dtype, _np.floating) and arr.max() <= 1.01:
        arr = arr * 255.0
    return _np.clip(arr, 0, 255).astype(_np.uint8, copy=False)


def maybe_load_lora(pipe, lora_path: str):
    if not lora_path:
        return
    if not os.path.isdir(lora_path):
        print(f"[LoRA] WARNING: '{lora_path}' is not a directory saved via save_attn_procs(). Continuing base-only.")
        return
    try:
        pipe.transformer.load_attn_procs(lora_path)
        print(f"[LoRA] Loaded attention processors from: {lora_path}")
    except Exception as e:
        print(f"[LoRA] WARNING: failed to load from '{lora_path}': {e}. Continuing base-only.")


def enable_memory_savers(pipe):
    for fn in ("enable_attention_slicing",):
        try:
            getattr(pipe, fn)()
        except Exception:
            pass
    # VAE tiling/slicing (pipeline may or may not expose .vae)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    # xFormers if present
    try:
        import xformers  # noqa: F401
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    except Exception:
        pass


def extract_latents_or_frames(output):
    """
    Normalize Diffusers outputs into either:
      ("latents", 5D tensor)  or  ("frames", list/np/tensor already decoded)
    Handles: dicts, ImagePipelineOutput, tuples/lists, raw np.ndarray, PIL.Image lists.
    """
    import numpy as _np
    import torch as _torch
    from PIL import Image as _PILImage

    # 1) None?
    if output is None:
        raise RuntimeError("Pipeline returned None.")

    # 2) Common object/dict forms
    for key in ("latents", "frames", "videos", "images"):
        if hasattr(output, key):
            val = getattr(output, key)
            if key == "latents":
                return "latents", val
            return "frames", val
        if isinstance(output, dict) and key in output:
            val = output[key]
            if key == "latents":
                return "latents", val
            return "frames", val

    # 3) Raw numpy array directly
    if isinstance(output, _np.ndarray):
        # Expect shape (T, H, W, C) or (B, T, H, W, C). We'll handle downstream.
        return "frames", output

    # 4) Tuple/list forms from return_dict=False
    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        # a) Tuple containing a single ndarray or frames list
        if len(output) == 1 and (isinstance(first, _np.ndarray) or isinstance(first, list)):
            return "frames", first
        # b) List/tuple of PIL images
        if isinstance(first, _PILImage.Image):
            return "frames", list(output)
        # c) List/tuple of ndarrays / tensors
        if isinstance(first, (_np.ndarray, _torch.Tensor, dict)):
            return "frames", list(output)

    # 5) Last resort: if it's a torch.Tensor with 5D video
    if isinstance(output, _torch.Tensor) and output.dim() == 5:
        return "latents", output

    # Debug breadcrumb so we can see what shape hit us
    try:
        print("[extract] Unrecognized output type:", type(output))
        print("[extract] dir(output):", dir(output))
    except Exception:
        pass
    raise RuntimeError("Could not find latents or frames in pipeline output.")


@torch.no_grad()
def decode_in_time_chunks(vae, latents: torch.Tensor, t_chunk: int = 3):
    """
    latents: (B=1, C, T, H, W) — decode in small T slices to keep VRAM low.
    Returns list of frames (HWC uint8).
    """
    if vae is None:
        raise RuntimeError("Pipeline VAE is missing; cannot decode latents.")

    # WAN / Diffusers VAEs store the upscaling factor here
    scale = float(getattr(getattr(vae, "config", object()), "scaling_factor", 1.0))

    frames_out = []
    _, _, T, _, _ = latents.shape
    for t0 in range(0, T, t_chunk):
        t1 = min(T, t0 + t_chunk)
        slab = latents[:, :, t0:t1]                       # (1, C, t, H, W)
        slab = slab * scale                               # << crucial: rescale latents

        # Many VAEs return sample in [-1, 1]
        decoded = vae.decode(slab, return_dict=False)[0]  # (1, t, H, W, 3) or (1, 3, t, H, W) depending on VAE

        if isinstance(decoded, torch.Tensor):
            # Normalize to [0, 255] safely
            # Try the common [-1,1] -> [0,1] path; if it's already [0,1], this still behaves.
            img = (decoded.clamp(-1, 1) + 1.0) * 0.5
            img = (img * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            # Fallback for odd return types; best-effort clamp/scale
            import numpy as _np
            arr = _np.asarray(decoded)
            if arr.dtype.kind == "f":
                # handle [-1,1] or [0,1]
                if arr.min() < 0.0:
                    arr = (arr.clip(-1, 1) + 1.0) * 0.5
                arr = (arr * 255.0)
            img = _np.clip(arr, 0, 255).astype(_np.uint8, copy=False)

        # Ensure time dimension is [t] in position 1
        # Expecting img shape (1, t, H, W, C)
        if img.ndim == 5 and img.shape[1] == (t1 - t0):
            for k in range(img.shape[1]):
                frames_out.append(img[0, k])
        else:
            # Last-resort heuristics
            from numpy import moveaxis
            if img.ndim == 5 and img.shape[2] == (t1 - t0):
                img = moveaxis(img, 2, 1)
                for k in range(img.shape[1]):
                    frames_out.append(img[0, k])
            elif img.ndim == 4:
                for k in range(img.shape[0]):
                    frames_out.append(img[k])
            else:
                raise RuntimeError(f"Unexpected decoded shape {img.shape}")

        del slab, decoded, img
        torch.cuda.empty_cache()
        gc.collect()
    return frames_out


def payload_to_frame_list(payload):
    import numpy as _np, torch as _torch

    def _npify(a):
        if isinstance(a, _torch.Tensor):
            a = a.detach().cpu()
        return _np.asarray(a)

    def _to_list(frames):
        if frames.dtype.kind == "f" and frames.max() <= 1.01:
            frames = frames * 255.0
        frames = _np.clip(frames, 0, 255).astype(_np.uint8, copy=False)
        return [frames[t] for t in range(frames.shape[0])]

    if isinstance(payload, list):
        out = []
        for fr in payload:
            fr = _npify(fr)
            if fr.ndim == 2:
                fr = _np.repeat(fr[..., None], 3, axis=2)
            if fr.ndim == 3 and fr.shape[0] in (1, 3, 4) and fr.shape[-1] not in (1, 3, 4):
                fr = _np.transpose(fr, (1, 2, 0))
            if fr.ndim != 3:
                fr = _np.zeros((8, 8, 3), _np.uint8)
            if fr.dtype.kind == "f" and fr.max() <= 1.01:
                fr = (fr * 255.0)
            fr = _np.clip(fr, 0, 255).astype(_np.uint8, copy=False)
            out.append(fr)
        return out

    arr = _npify(payload)

    if arr.ndim == 5:
        ch_ax = [i for i, s in enumerate(arr.shape) if s in (1, 3, 4)][-1]
        arr = _np.moveaxis(arr, ch_ax, -1)
        arr = arr[0]
    elif arr.ndim == 4:
        sizes = list(arr.shape)
        ch_candidates = [i for i, s in enumerate(sizes) if s in (1, 3, 4)]
        ch_ax = ch_candidates[-1] if ch_candidates else None
        if ch_ax is not None and ch_ax != 3:
            arr = _np.moveaxis(arr, ch_ax, 3)
        if arr.shape[0] not in (1, arr.shape[1], arr.shape[2]):   # (T,H,W,C)
            pass
        elif arr.shape[2] not in (arr.shape[0], arr.shape[1]):    # (H,W,T,C)
            arr = _np.moveaxis(arr, 2, 0)
        elif arr.shape[1] not in (arr.shape[0], arr.shape[2]):    # (H,T,W,C)
            arr = _np.moveaxis(arr, 1, 0)
        else:
            if arr.shape[2] <= 8 and arr.shape[0] >= 9:
                arr = _np.moveaxis(arr, 2, 0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = _np.transpose(arr, (1, 2, 0))
        arr = arr[None]
    else:
        raise ValueError(f"Unexpected payload shape: {arr.shape}")

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = _np.repeat(arr, 3, axis=3)
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4, 1):
        arr = _np.zeros((1, 8, 8, 3), dtype=_np.uint8)

    return _to_list(arr)


def autocast_ctx(device, dtype):
    # kept for future use if you want context managers externally
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.cpu.amp.autocast(dtype=torch.float32, enabled=False)


def denoise_request(
    pipe,
    prompt,
    neg,
    frames,   # int (count) for WAN OR list-of-images for FramePack
    size,     # int (square side length)
    steps,
    cfg_scale,
    generator,
    device,
    dtype,
    want_latents=False,
    sampling_type=None,
):
    # Treat size as square (H=W=size)
    h = w = size

    # Base kwargs shared
    kwargs = dict(
        prompt=prompt,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    if sampling_type is not None:
        kwargs["sampling_type"] = sampling_type

    if neg is not None:
        # Some video pipelines ignore this; harmless if unsupported
        kwargs["negative_prompt"] = neg

    if isinstance(pipe, HunyuanVideoFramepackPipeline):
        # FramePack: we want decoded frames; enforce ndarray return
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Framepack backend requires at least one init frame. Provide frames or inject a dummy frame.")
        first = frames[0]
        last = frames[-1] if len(frames) > 1 else None
        kwargs["image"] = first
        if last is not None:
            kwargs["last_image"] = last
        kwargs["num_frames"] = len(frames)
        # Ask for raw numpy and non-dict return to normalize outputs
        kwargs["return_dict"] = False
        kwargs["output_type"] = "np"
    else:
        # WAN: frames is an integer count
        if not isinstance(frames, int):
            raise ValueError("WAN backend expects `frames` as an int (frame count).")
        kwargs["num_frames"] = frames
        if want_latents:
            kwargs["output_type"] = "latent"
            kwargs["return_dict"] = True
        else:
            kwargs["output_type"] = "np"
            kwargs["return_dict"] = False

    out = pipe(**kwargs)
    return out


@torch.no_grad()
def run_one(
    pipe,
    prompt,
    neg,
    frames,      # WAN: int count; FramePack: list-of-images or None/int (we'll build)
    size,        # int (square)
    steps,
    cfg_scale,
    generator,
    device,
    dtype,
    t_chunk=3,
):
    """
    FramePack behavior:
      - If `frames` is a list/tuple, it's used as-is (expects [first, last] or list of frames).
      - If not provided as list/tuple:
         1) If env FRAMEPACK_INIT_DIR is set, loads first/last image from that dir.
         2) Else, if helper `make_keyframes_from_prompt` exists, uses it to T2I-generate two keyframes.
         3) If only one keyframe is available, synthesize a second by small rotation/shift.
         4) Otherwise, raises a clear error (no mid-gray dummies anymore).
    WAN behavior unchanged.
    """
    import numpy as _np

    is_framepack = isinstance(pipe, HunyuanVideoFramepackPipeline)

    def _to_uint8_hwc(arr):
        a = _np.asarray(arr)
        if a.ndim == 2:
            a = _np.repeat(a[..., None], 3, axis=2)
        if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
            a = _np.transpose(a, (1, 2, 0))  # CHW -> HWC
        if a.dtype.kind == "f":
            # handle [0,1] or [-1,1]
            if a.min() < 0:
                a = (a.clip(-1, 1) + 1.0) * 0.5
            a = a * 255.0
        return _np.clip(a, 0, 255).astype(_np.uint8, copy=False)

    def _load_keyframes_from_dir(init_dir, size_px):
        from pathlib import Path
        from PIL import Image
        p = Path(init_dir)
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        paths = sorted([q for q in p.iterdir() if q.suffix.lower() in exts])
        if not paths:
            raise ValueError(f"No keyframe images found in {init_dir}")
        def _read(path):
            im = Image.open(path).convert("RGB").resize((size_px, size_px))
            return _np.asarray(im)
        first = _read(paths[0])
        last  = _read(paths[-1]) if len(paths) > 1 else None
        return [first] if last is None else [first, last]

    def _synthesize_second(img):
        # light rotation + slight translation using PIL, no OpenCV needed
        from PIL import Image
        H, W = img.shape[:2]
        im = Image.fromarray(img)
        im = im.rotate(2.0, resample=Image.BILINEAR, expand=False)
        # translate by a couple of pixels
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(im, (2, 0))
        return _np.asarray(canvas)

    # ---- Build frames for FramePack if needed ----
    if is_framepack:
        if not isinstance(frames, (list, tuple)):
            kfs = None

            # 1) Try env-provided directory
            init_dir = os.environ.get("FRAMEPACK_INIT_DIR", "").strip()
            if init_dir:
                kfs = _load_keyframes_from_dir(init_dir, size)

            # 2) If still none, try T2I helper if user defined it
            if kfs is None and "make_keyframes_from_prompt" in globals():
                try:
                    kfs = list(globals()["make_keyframes_from_prompt"](prompt, size, dtype, device))
                except Exception as e:
                    print(f"[FramePack] T2I keyframe generation failed: {e}")

            # 3) If still none, hard fail (no gray dummies)
            if not kfs:
                raise ValueError(
                    "FramePack requires real keyframes. "
                    "Provide a frames list, set FRAMEPACK_INIT_DIR to a folder with images, "
                    "or define make_keyframes_from_prompt(...) to auto-generate."
                )

            # 4) If only one, synthesize a second
            if len(kfs) == 1:
                kfs = [kfs[0], _synthesize_second(_to_uint8_hwc(kfs[0]))]

            # Ensure uint8 HWC
            frames = [_to_uint8_hwc(k) for k in kfs]
        else:
            # Ensure dtype/shape for user-provided frames
            frames = [_to_uint8_hwc(k) for k in frames]

    # ---- Denoise request + extraction ----
    try:
        out = denoise_request(
            pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype,
            want_latents=(not is_framepack)
        )
        kind, payload = extract_latents_or_frames(out)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        size2 = max(256, (size // 2) // 8 * 8)
        if is_framepack:
            frames2 = frames  # list stays as-is
        else:
            frames2 = max(9, (frames if isinstance(frames, int) else 49) - 4)
            frames2 = round_wan_frames(frames2)
        steps2 = max(12, steps // 2)
        print(f"[OOM@denoise] Retrying with size {size}->{size2}, frames {frames}->{frames2}, steps {steps}->{steps2}")
        out = denoise_request(
            pipe, prompt, neg, frames2, size2, steps2, cfg_scale, generator, device, dtype,
            want_latents=(not is_framepack)
        )
        kind, payload = extract_latents_or_frames(out)
        size, frames, steps = size2, frames2, steps2
    except Exception as e:
        print(f"[Warn] First request failed ({e}). Retrying with decoded frames path.")
        out = denoise_request(
            pipe, prompt, neg, frames, size, steps, cfg_scale, generator, device, dtype,
            want_latents=False
        )
        kind, payload = extract_latents_or_frames(out)

    # ---- Decode / normalize to frame list ----
    if kind == "latents":
        latents = payload  # (1, C, T, H, W)
        try:
            frames_rgb = decode_in_time_chunks(getattr(pipe, "vae", None), latents, t_chunk=t_chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print("[OOM@decode] Retrying VAE decode with smaller time chunks (t_chunk=2)")
            frames_rgb = decode_in_time_chunks(getattr(pipe, "vae", None), latents, t_chunk=2)
    else:
        frames_rgb = payload_to_frame_list(payload)

    del out
    torch.cuda.empty_cache()
    gc.collect()
    return frames_rgb


def load_pipeline(cfg, backend, dtype, device):
    if backend == "wan":
        pipe = DiffusionPipeline.from_pretrained(
            cfg["model"]["base_ckpt"], dtype=dtype, variant=None
        ).to(device)
        return pipe, "wan"

    # --- FRAMEPACK: match official example repos & dtypes ---
    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        cfg["model"]["framepack_transformer"], torch_dtype=torch.bfloat16
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(
        cfg["model"]["framepack_flux"], subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        cfg["model"]["framepack_flux"], subfolder="image_encoder", torch_dtype=torch.float16
    )

    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        cfg["model"]["framepack_ckpt"],
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    ).to(device)

    # memory helpers
    try: pipe.enable_model_cpu_offload()
    except: pass
    try: pipe.vae.enable_tiling()
    except: pass

    return pipe, "framepack"


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", choices=["wan", "framepack"], default="wan")
    ap.add_argument("--init_dir", type=str, default="", help="Directory with keyframe images (first/last).")
    ap.add_argument("--t2i_model", type=str, default="", help="Diffusers T2I model id for auto keyframes")
    args = ap.parse_args()

    # 1) Load cfg FIRST, then read from it. Do not touch cfg before this.
    cfg = load_cfg(args.config)

    # 2) Decide how FramePack will get keyframes
    #    Expose to run_one via globals since you referenced globals() there.
    if args.init_dir:
        init_frames_source = "dir"
    elif args.backend == "framepack":
        init_frames_source = "t2i"
    else:
        init_frames_source = None

    if not args.t2i_model:
        # allow YAML override
        args.t2i_model = cfg.get("model", {}).get("framepack_t2i", "black-forest-labs/FLUX.1-dev")

    globals()["init_frames_source"] = init_frames_source
    globals()["args"] = args  # used inside run_one

    # ---- device/dtype and sampler/dataset config (unchanged) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = pick_dtype(cfg)

    seed       = int(cfg.get("sampler", {}).get("seed", 42))
    frames     = int(cfg["sampler"]["frames"])
    fps        = int(cfg["sampler"]["fps"])
    size       = int(cfg["sampler"]["size"])
    steps      = int(cfg["sampler"]["steps"])
    cfg_scale  = float(cfg["sampler"]["cfg_scale"])
    neg_prompt = cfg["sampler"].get("negative_prompt", None)
    t_chunk    = int(cfg.get("sampler", {}).get("vae_t_chunk", 3))

    classes   = cfg["dataset"]["classes"]
    per_class = int(cfg["dataset"]["per_class"])
    sampling_type = cfg["sampler"].get("sampling_type", None)

    # ---- load pipeline and generate (unchanged) ----
    backend = args.backend
    pipe, backend = load_pipeline(cfg, backend, dtype, device)
    enable_memory_savers(pipe)

    if backend == "wan":
        new_frames = round_wan_frames(frames)
        if new_frames != frames:
            print(f"[Note] Rounded frames {frames} → {new_frames} to satisfy WAN’s requirement.")
            frames = new_frames
        lora_path = cfg["model"].get("lora_path", "")
        maybe_load_lora(pipe, lora_path)

    out_root = Path(cfg["dataset"]["out_dir"]) / backend
    out_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating videos with backend: {backend} …")
    for cls in classes:
        cdir = out_root / cls
        cdir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(per_class), desc=f"{cls:>12}"):
            prompt = f"a person performing {cls}"
            frames_rgb = run_one(
                pipe=pipe,
                prompt=prompt,
                neg=neg_prompt,
                frames=frames,      # WAN: int; FramePack: our run_one builds real keyframes if needed
                size=size,
                steps=steps,
                cfg_scale=cfg_scale,
                generator=generator,
                device=device,
                dtype=dtype,
                t_chunk=t_chunk,
            )
            path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(path, [to_hwc_uint8(fr) for fr in frames_rgb], fps=fps, macro_block_size=1)
            del frames_rgb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("All videos generated successfully.")


if __name__ == "__main__":
    main()
