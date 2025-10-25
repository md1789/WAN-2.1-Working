# generate_videos.py
import os, argparse, random
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["dataset"]["classes"]
    per_class = int(cfg["dataset"]["per_class"])
    out_dir = Path(cfg["dataset"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = int(cfg["sampler"]["frames"])
    fps = int(cfg["sampler"]["fps"])
    size = int(cfg["sampler"]["size"])

    # Placeholder generator: produces noise videos to test pipeline wiring
    for cls in classes:
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            vid = []
            seed = hash((cls, i)) % (2**32)
            rng = np.random.RandomState(seed)
            base = rng.rand(size, size, 3)
            for t in range(frames):
                frame = (base + 0.1 * rng.randn(size, size, 3)).clip(0,1)
                vid.append((frame * 255).astype(np.uint8))
            path = cdir / f"{cls}_{i:02d}.mp4"
            imageio.mimwrite(path, vid, fps=fps)
    print("Generated placeholder videos. Swap with real sampler calls.")

if __name__ == "__main__":
    main()
