# WAN 2.1 LoRA — Human Activity Video Generation (Clean Start)

This repository is re-initialized to **avoid CLI patch chaos** and run cleanly on **RunPod**.

## What this does
- Trains **LoRA adapters** on **WAN 2.1 (1.3G)** for *human activity* motions.
- Generates **10 videos per class** for grading.
- Provides a minimal, swappable scaffold so you can plug in your WAN 2.1 code without fighting deps.

## Quickstart (RunPod, CUDA 12)
```bash
pip install -r requirements.txt

# 1) Get data
python scripts/prepare_kaggle.py --dataset sharjeelmazhar/human-activity-recognition-video-dataset --out data/har

# 2) Train LoRA (replace dummy with WAN integration later)
python train_lora.py --config configs/lora_wan21_har.yaml

# 3) Generate videos (placeholder sampler; replace with WAN sampler)
python generate_videos.py --config configs/infer_har.yaml
```

## Next steps to wire WAN
- Load WAN 2.1 1.3G backbone into `train_lora.py` and inject LoRA with **PEFT**.
- Replace dummy dataset collation with frame decoding using **decord**.
- Swap `generate_videos.py` placeholder with WAN sampler calls (EMA weights).

See `ANALYSIS.md` for the grading-focused analysis template.
