import argparse, os, math, torch, numpy as np
from pathlib import Path
from torchvision import transforms
import imageio.v3 as iio

# ---- I3D backbone (Kinetics-400) ----
# Using torch.hub weights for simplicity; swap to your preferred I3D loader if you like.
i3d = torch.hub.load("moabitcoin/ig65m-pytorch", "i3d_resnet50", pretrained=True)  # RGB stream
i3d.eval().cuda().requires_grad_(False)

# Standardize to 224x224, T frames
def load_clip(path, T=32, fps=8):
    # Read, uniformly sample/loop to T, center-crop & resize -> (T,3,224,224), 0..1
    vid = iio.imread(path, index=None)  # (num_frames,H,W,3), uint8
    if vid.ndim == 3:  # single frame
        vid = vid[None]
    num = len(vid)
    idx = np.linspace(0, max(1, num-1), T).round().astype(int)
    clip = vid[idx]
    x = torch.from_numpy(clip).float()/255.0  # (T,H,W,3)
    x = x.permute(0,3,1,2)  # (T,3,H,W)
    tfm = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
    ])
    x = tfm(x)  # (T,3,224,224)
    return x

@torch.no_grad()
def i3d_embed(x):
    # x: (B,T,3,224,224) in [0,1]
    B,T,_,_,_ = x.shape
    # I3D often expects 25fps-ish; we’re close enough for FVD features
    x = x.cuda()
    feats = i3d.extract_features(x)  # (B, C, t', h', w') or a pooled vector depending on hub impl
    if feats.dim() > 2:
        feats = feats.mean(dim=[2,3,4])  # global average pool
    return feats.float().cpu().numpy()

def stats_from_dir(d, T=32):
    paths = [str(p) for p in Path(d).glob("**/*.mp4")]
    all_feats = []
    bs = 8
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i+bs]:
            x = load_clip(p, T=T)  # (T,3,224,224)
            batch.append(x[None])
        if not batch: break
        X = torch.cat(batch, dim=0)  # (B, T, 3, 224, 224)
        feats = i3d_embed(X)
        all_feats.append(feats)
    F = np.concatenate(all_feats, axis=0)
    mu = F.mean(axis=0)
    sigma = np.cov(F, rowvar=False)
    return mu, sigma

def frechet(mu1, sig1, mu2, sig2, eps=1e-6):
    diff = mu1 - mu2
    # sqrtm via eigen (stable & fast for covs)
    vals1, vecs1 = np.linalg.eigh(sig1)
    vals2, vecs2 = np.linalg.eigh(sig2)
    # product sqrt
    sqrt1 = (vecs1 * np.sqrt(np.maximum(vals1, eps))) @ vecs1.T
    prod  = sqrt1 @ sig2 @ sqrt1
    valsP, _ = np.linalg.eigh(prod)
    tr_sqrt = np.sum(np.sqrt(np.maximum(valsP, eps)))
    return diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2.0*tr_sqrt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)       # e.g., your HAR real videos
    ap.add_argument("--gen_dir", required=True)        # e.g., outputs/samples/walking + running + ...
    ap.add_argument("--frames", type=int, default=32)  # FVD uses fixed T
    args = ap.parse_args()

    mu_r, sg_r = stats_from_dir(args.real_dir, T=args.frames)
    mu_g, sg_g = stats_from_dir(args.gen_dir,  T=args.frames)
    score = frechet(mu_r, sg_r, mu_g, sg_g)
    print(f"FVD = {score:.2f}")
