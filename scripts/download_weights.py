"""Download and verify ALL pretrained weights used by the project, up front.

Weights fetched (nothing is trained in this project):

| Component            | Source                                                     | Used by            |
|----------------------|------------------------------------------------------------|--------------------|
| AdaIN decoder        | github.com/naoto0804/pytorch-AdaIN release v0.0.0           | Method 2 / 3       |
| AdaIN normalised VGG | same release (vgg_normalised.pth)                           | Method 2 / 3       |
| torchvision VGG-19   | torchvision VGG19_Weights.IMAGENET1K_V1                     | Method 1 (Gatys)   |
| LPIPS (AlexNet)      | `lpips` package linear layers + torchvision AlexNet         | evaluation         |
| CLIP ViT-B/32        | open_clip `ViT-B-32` / `laion2b_s34b_b79k`                  | evaluation         |
| DeepLabV3            | torchvision deeplabv3_mobilenet_v3_large (COCO-VOC labels)  | semantic stretch   |

Usage:  python scripts/download_weights.py [--skip-optional]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "weights"
ADAIN_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/{}"
ADAIN_FILES = {  # filename -> expected size in bytes (from the GitHub release)
    "decoder.pth": 14_023_458,
    "vgg_normalised.pth": 80_102_481,
}
CLIP_MODEL, CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def _fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}\n  -> Check your internet connection / proxy and re-run "
          f"`python scripts/download_weights.py`.", file=sys.stderr)
    sys.exit(1)


def _download(url: str, dst: Path, expected_size: int | None = None) -> None:
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size):
        print(f"  [ok] {dst.relative_to(ROOT)} (cached)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  [..] downloading {url}")
    try:
        from tqdm import tqdm
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("content-length", 0)) or None
            with tqdm(total=total, unit="B", unit_scale=True, leave=False) as bar:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:  # network failure
        tmp.unlink(missing_ok=True)
        _fail(f"Download failed for {url}: {e}")
    if expected_size is not None and tmp.stat().st_size != expected_size:
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        _fail(f"Size mismatch for {dst.name}: got {size}, expected {expected_size}")
    tmp.rename(dst)
    print(f"  [ok] {dst.relative_to(ROOT)}")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-optional", action="store_true",
                    help="only fetch weights needed for the core pipeline (AdaIN + VGG-19)")
    args = ap.parse_args()

    import torch
    hub = WEIGHTS / "torch_hub"
    torch.hub.set_dir(str(hub))
    os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf"))

    print("1) AdaIN encoder + decoder (pretrained, naoto0804/pytorch-AdaIN)")
    for name, size in ADAIN_FILES.items():
        _download(ADAIN_URL.format(name), WEIGHTS / "adain" / name, size)
    # verify they load into the exact architectures
    from src.models import AdaINDecoder, AdaINEncoder
    AdaINEncoder(); AdaINDecoder()
    for name in ADAIN_FILES:
        print(f"       sha256[:16] {name}: {_sha256(WEIGHTS / 'adain' / name)}")
    print("  [ok] AdaIN state dicts load into the matching architectures")

    print("2) torchvision VGG-19 (ImageNet) for Gatys")
    import torchvision
    try:
        torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    except Exception as e:
        _fail(f"torchvision VGG-19 download failed: {e}")
    print("  [ok] VGG-19")

    if args.skip_optional:
        print("Skipping optional weights (LPIPS, CLIP, DeepLabV3).")
        return

    print("3) LPIPS (AlexNet backbone)")
    try:
        import lpips
        lpips.LPIPS(net="alex", verbose=False)
    except Exception as e:
        _fail(f"LPIPS weights failed: {e}")
    print("  [ok] LPIPS")

    print(f"4) CLIP {CLIP_MODEL} ({CLIP_PRETRAINED})")
    try:
        import open_clip
        open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED,
                                              cache_dir=str(WEIGHTS / "clip"))
    except Exception as e:
        _fail(f"CLIP weights failed: {e}")
    print("  [ok] CLIP")

    print("5) DeepLabV3-MobileNetV3 (semantic sky/foreground stretch goal)")
    try:
        torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    except Exception as e:
        _fail(f"DeepLabV3 weights failed: {e}")
    print("  [ok] DeepLabV3")
    print("\nAll weights downloaded and verified.")


if __name__ == "__main__":
    main()
