# scripts/prepare_kaggle.py
import argparse, os, shutil, subprocess, sys, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g., sharjeelmazhar/human-activity-recognition-video-dataset")
    ap.add_argument("--out", default="data/har")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Expecting kaggle CLI configured on RunPod
    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # unzip any archives
    for f in out.glob("*.zip"):
        subprocess.check_call(["unzip", "-o", str(f), "-d", str(out)])
    print("Done. Data at", out)

if __name__ == "__main__":
    main()
