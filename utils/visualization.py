import imageio
import numpy as np

def write_video(frames, out_path: str, fps: int = 15):
    # frames: list[np.ndarray(H,W,3)]
    imageio.mimwrite(out_path, [np.asarray(f) for f in frames], fps=fps, quality=6)
