# models/WAN.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WANConfig:
    base_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    output_dir: str = "outputs/wan_lora"
    dp_path: str = "third_party/diffusion-pipe"
    config_path: str = "configs/my_wan21.toml"

def validate_structure(cfg: WANConfig) -> None:
    assert Path(cfg.dp_path).exists(), f"Missing diffusion-pipe at {cfg.dp_path}"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.config_path).parent.mkdir(parents=True, exist_ok=True)
