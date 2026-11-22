"""Shared utilities: device selection, seeding, image I/O, resizing and colour-space helpers.

Conventions used throughout the project
---------------------------------------
* The *internal* image format is a NumPy ``float32`` array of shape ``(H, W, 3)``,
  **RGB** channel order, values in ``[0, 1]``.
* OpenCV reads/writes **BGR**, so every OpenCV boundary converts explicitly.
* Tensors fed to networks are ``(1, 3, H, W)`` float32; each network applies its
  *own* normalisation (see ``src/models.py``).
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np
import torch

PathLike = Union[str, os.PathLike]

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_INPUT_PIXELS = 40_000_000  # refuse absurdly large inputs (≈ 40 MP)

logger = logging.getLogger("mood_transfer")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a simple console logger once and return the project logger."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    return logger


# --------------------------------------------------------------------------------------
# Device & reproducibility
# --------------------------------------------------------------------------------------
def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU.

    Args:
        prefer: optional explicit device string ("cpu", "cuda", "mps").
    """
    if prefer:
        if prefer == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        if prefer == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_size(device: torch.device) -> int:
    """Default working resolution (long side): 512 on a GPU, 384 on CPU."""
    return 384 if device.type == "cpu" else 512


def set_seed(seed: int = 0) -> None:
    """Fix all random seeds (Python, NumPy, PyTorch) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory(device: torch.device) -> None:
    """Release cached GPU memory (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def sync(device: torch.device) -> None:
    """Synchronise the device so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------------------
def load_image(path: PathLike, max_side: int | None = None) -> np.ndarray:
    """Load an image file as float32 RGB in [0, 1].

    Handles grayscale (replicated to 3 channels), alpha (dropped) and 16-bit images.

    Raises:
        FileNotFoundError / ValueError for missing or undecodable files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not decode image: {path}")
    return to_float_rgb(raw, bgr=True, max_side=max_side)


def to_float_rgb(arr: np.ndarray, bgr: bool = False, max_side: int | None = None) -> np.ndarray:
    """Convert any uint8/uint16/float array (gray, RGB, RGBA, BGR) to float32 RGB [0,1]."""
    if arr is None:
        raise ValueError("Empty image")
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Empty image")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype == np.uint8:
        img = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        img = arr.astype(np.float32) / 65535.0
    else:
        img = arr.astype(np.float32)
        if img.max() > 1.0 + 1e-3:
            img = img / 255.0
    if bgr:
        img = img[..., ::-1]
    if img.shape[0] * img.shape[1] > MAX_INPUT_PIXELS:
        raise ValueError(f"Image too large ({img.shape[1]}x{img.shape[0]}); max {MAX_INPUT_PIXELS} px.")
    img = np.ascontiguousarray(np.clip(img, 0.0, 1.0))
    if max_side is not None:
        img = resize_max_side(img, max_side)
    return img


def save_image(img: np.ndarray, path: PathLike) -> Path:
    """Clamp to [0,1], convert RGB->BGR uint8 and write to disk (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if not cv2.imwrite(str(path), out[..., ::-1]):
        raise IOError(f"Failed to write {path}")
    return path


def list_images(folder: PathLike) -> list[Path]:
    """Sorted list of image files in a folder (non-recursive)."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint8 [0,255] (rounded)."""
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Resizing / padding
# --------------------------------------------------------------------------------------
def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Resize so the longer side equals ``max_side`` (aspect ratio preserved).

    Uses INTER_AREA for downscaling (anti-aliased) and INTER_CUBIC for upscaling.
    """
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return img
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (nw, nh), interpolation=interp), 0, 1).astype(np.float32)


def resize_to(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize to an exact (H, W)."""
    h, w = hw
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_AREA if h * w < img.shape[0] * img.shape[1] else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (w, h), interpolation=interp), 0, 1).astype(np.float32)


def pad_to_multiple(t: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad an (N,C,H,W) tensor so H and W are multiples of ``multiple``.

    The AdaIN encoder max-pools 3 times and the decoder upsamples ×8, so odd sizes
    would otherwise produce an off-by-a-few output. Returns the padded tensor and the
    original (H, W) for cropping back.
    """
    h, w = t.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        mode = "reflect" if (h > ph and w > pw) else "replicate"
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode=mode)
    return t, (h, w)


# --------------------------------------------------------------------------------------
# NumPy <-> Torch
# --------------------------------------------------------------------------------------
def np_to_tensor(img: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """(H,W,3) float [0,1] -> (1,3,H,W) float tensor on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float().to(device)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) tensor -> (H,W,3) float32 [0,1] (clamped)."""
    return t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# Colour-space helpers (float RGB [0,1] in/out)
# --------------------------------------------------------------------------------------
def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> CIE L*a*b* (L in [0,100], a/b roughly [-128,127]).

    OpenCV's float32 path implements the exact CIE formulas (sRGB gamma, D65 white).
    Lab is (approximately) perceptually uniform and decorrelates lightness from
    chroma, which is why colour-transfer statistics are computed there.
    """
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* -> RGB [0,1] (clipped to gamut)."""
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> YCbCr (ITU-R BT.601, full range, all channels in [0,1], chroma centred 0.5)."""
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)
    ycc = img.astype(np.float32) @ m.T
    ycc[..., 1:] += 0.5
    return ycc


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycbcr`."""
    y = ycc[..., 0]
    cb = ycc[..., 1] - 0.5
    cr = ycc[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """BT.601 luma Y in [0,1]."""
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend: (1-alpha)*a + alpha*b."""
    return np.clip((1.0 - alpha) * a + alpha * b, 0, 1).astype(np.float32)


def make_grid_row(images: Iterable[np.ndarray], height: int = 256, gap: int = 4) -> np.ndarray:
    """Concatenate images horizontally at a common height (white gap between)."""
    row = []
    for im in images:
        h, w = im.shape[:2]
        im = resize_to(im, (height, max(1, round(w * height / h))))
        row.append(im)
        row.append(np.ones((height, gap, 3), np.float32))
    return np.concatenate(row[:-1], axis=1)
