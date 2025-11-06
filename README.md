# 🎬 WAN-2.1 LoRA + FramePack Video Generation & FVD Evaluation

This repository provides a **unified pipeline** for training and evaluating
LoRA adapters on the **WAN 2.1 text-to-video model**, and generating or evaluating videos using either **WAN** or **FramePack** (HunyuanVideo) backends.

Built for research in **Human Action Recognition (HAR)** and **video diffusion model evaluation**, this project allows you to:

- Train custom **LoRA adapters** on your dataset  
- Generate videos with either **WAN 2.1** *or* **FramePack**
- Compute **Fréchet Video Distance (FVD)** between generated and reference videos

---

## Features
| Capability | Description |
|-------------|-------------|
| **WAN 2.1** | Text-to-video generation via [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) |
| **FramePack** | [HunyuanVideoFramepackPipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/framepack) backend for fast, memory-efficient video diffusion |
| **LoRA Training** | Fine-tune WAN attention layers using `train_lora.py` |
| **Video Generation** | Flexible, backend-agnostic script: `generate_videos.py` |
| **Metrics / FVD** | Evaluate realism vs. reference clips using `metrics/fvd_eval.py` |
| **Chunked Decoding** | Automatically decodes latents in time-chunks to reduce VRAM |
| **Colab-Friendly** | Fully runnable on Google Colab with A100 / T4 GPUs |

---

## Installation
```bash
git clone https://github.com/md1789/WAN-2.1-Working.git
cd WAN-2.1-Working
pip install -r requirements.txt
```

## Video Generation
WAN 2.1 Backend
```bash
python generate_videos.py --config configs/infer_har.yaml --backend wan
```
FramePack Backend
```bash
python generate_videos.py --config configs/infer_har_framepack.yaml --backend framepack
```

Outputs ->
outputs/samples/wan/<class>/<name>.mp4
outputs/samples/framepack/<class>/<name>.mp4


## FVD Evaluation
```bash
python metrics/fvd_eval.py \
  --ref_dir "data/har/Human Activity Recognition - Video Dataset" \
  --gen_dir "outputs/samples/wan" \
  --fps 8 --num_frames 32 --resolution 448 \
  --recursive \
  --write_csv "outputs/metrics/fvd_all.csv"
```

## LORA Training
```bash
python train_lora.py --config configs/train_lora.yaml
```

Outputs -> outputs/lora_har/lora_ema_last


## Colab Quickstart
!nvidia-smi
!pip install -r requirements.txt
!python generate_videos.py --config configs/infer_har.yaml --backend wan
!python metrics/fvd_eval.py --ref_dir data/har/... --gen_dir outputs/samples/wan --fps 8 --num_frames 32

WAN-2.1-Working/
├── configs/
│   ├── infer_har.yaml
│   ├── infer_har_framepack.yaml
│   └── train_lora.yaml
├── train_lora.py
├── generate_videos.py
├── metrics/
│   └── fvd_eval.py
├── outputs/
│   ├── lora_har/
│   └── samples/
└── requirements.txt

## License
MIT License © 2025 Megan Diehl.
WAN 2.1 and FramePack models are governed by their respective Hugging Face licenses.