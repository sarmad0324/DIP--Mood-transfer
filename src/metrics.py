"""Evaluation metrics.

Content preservation (output vs content):
    * **SSIM** on luminance (structure, not colour — a mood change *should* change colour).
    * **LPIPS** (AlexNet, Zhang et al. 2018) — learned perceptual distance; lower = closer.
      Computed on the luminance replicated to 3 channels so it measures structure.
      ``lpips_rgb`` (full colour) is also stored for completeness.

Mood match (output vs reference):
    * **Lab mean/std distance** — ‖μ_out − μ_ref‖₂ + ‖σ_out − σ_ref‖₂ over (L*, a*, b*).
    * **Colour-histogram distances** — mean over Lab channels of the 1-D Wasserstein (EMD)
      distance, and Bhattacharyya distance between 3-D (8×8×8) RGB histograms.
    * **CLIP image–image cosine similarity** (ViT-B/32, LAION-2B) — semantic/mood
      similarity of the output to the reference; we also report the *gain* over the
      content's similarity to the same reference (ΔCLIP), which removes the bias of the
      content scene itself.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity

from .utils import WEIGHTS_DIR, get_device, luminance, resize_max_side, resize_to, rgb_to_lab


# --------------------------------------------------------------------------------------
# Content preservation
# --------------------------------------------------------------------------------------
def ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Wang et al. 2004) between the luminance channels (Gaussian window, σ=1.5)."""
    b = resize_to(b, a.shape[:2])
    return float(structural_similarity(luminance(a), luminance(b), data_range=1.0,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM averaged over RGB channels."""
    b = resize_to(b, a.shape[:2])
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


@lru_cache(maxsize=2)
def _lpips_model(device: str):
    import lpips
    torch.hub.set_dir(str(WEIGHTS_DIR / "torch_hub"))
    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_distance(a: np.ndarray, b: np.ndarray, luma: bool = True, device: str | None = None,
                   max_side: int = 512) -> float:
    """LPIPS distance (lower = more similar). Inputs float RGB [0,1]."""
    dev = device or str(get_device())
    a = resize_max_side(a, max_side)
    b = resize_to(b, a.shape[:2])
    if luma:
        a = np.repeat(luminance(a)[..., None], 3, 2)
        b = np.repeat(luminance(b)[..., None], 3, 2)
    ta = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1))[None].to(dev) * 2 - 1
    with torch.no_grad():
        return float(_lpips_model(dev)(ta, tb).item())


# --------------------------------------------------------------------------------------
# Mood match
# --------------------------------------------------------------------------------------
def lab_stats_distance(out: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Distance between Lab channel means and stds (Reinhard statistics)."""
    lo = rgb_to_lab(resize_max_side(out, 256)).reshape(-1, 3)
    lr = rgb_to_lab(resize_max_side(ref, 256)).reshape(-1, 3)
    dmean = float(np.linalg.norm(lo.mean(0) - lr.mean(0)))
    dstd = float(np.linalg.norm(lo.std(0) - lr.std(0)))
    return {"lab_mean_dist": dmean, "lab_std_dist": dstd, "lab_stat_dist": dmean + dstd}


def wasserstein_lab(out: np.ndarray, ref: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Mean 1-D Wasserstein distance over L*, a*, b* (subsampled pixels)."""
    rng = np.random.default_rng(seed)
    lo = rgb_to_lab(resize_max_side(out, 256)).reshape(-1, 3)
    lr = rgb_to_lab(resize_max_side(ref, 256)).reshape(-1, 3)
    lo = lo[rng.choice(len(lo), min(n, len(lo)), replace=False)]
    lr = lr[rng.choice(len(lr), min(n, len(lr)), replace=False)]
    return float(np.mean([wasserstein_distance(lo[:, c], lr[:, c]) for c in range(3)]))


def bhattacharyya_rgb(out: np.ndarray, ref: np.ndarray, bins: int = 8) -> float:
    """Bhattacharyya distance between normalised 3-D RGB histograms (0 = identical)."""
    def hist(x):
        h = cv2.calcHist([(np.clip(x, 0, 1) * 255).astype(np.uint8)], [0, 1, 2], None,
                         [bins] * 3, [0, 256] * 3)
        return cv2.normalize(h, h, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(hist(out), hist(ref), cv2.HISTCMP_BHATTACHARYYA))


# --------------------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------------------
class ClipScorer:
    """CLIP ViT-B/32 (open_clip, LAION-2B weights) image/text embeddings."""

    def __init__(self, device: str | None = None) -> None:
        import open_clip
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        self.device = device or str(get_device())
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(WEIGHTS_DIR / "clip"))
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._cache: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def embed_image(self, img: np.ndarray) -> torch.Tensor:
        from PIL import Image
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        e = self.model.encode_image(x)
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        t = self.tokenizer([text]).to(self.device)
        e = self.model.encode_text(t)
        return e / e.norm(dim=-1, keepdim=True)

    def image_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float((self.embed_image(a) * self.embed_image(b)).sum())

    def text_similarity(self, img: np.ndarray, text: str) -> float:
        return float((self.embed_image(img) * self.embed_text(text)).sum())


def all_metrics(content: np.ndarray, out: np.ndarray, ref: np.ndarray,
                clip: ClipScorer | None = None, clip_content_ref: float | None = None,
                device: str | None = None) -> dict[str, float]:
    """Compute every metric for one (content, output, reference) triple."""
    out = resize_to(out, content.shape[:2])
    m = {
        "ssim": ssim_luma(content, out),
        "lpips": lpips_distance(content, out, luma=True, device=device),
        "lpips_rgb": lpips_distance(content, out, luma=False, device=device),
        **lab_stats_distance(out, ref),
        "wasserstein_lab": wasserstein_lab(out, ref),
        "bhattacharyya": bhattacharyya_rgb(out, ref),
    }
    if clip is not None:
        m["clip_sim"] = clip.image_similarity(out, ref)
        if clip_content_ref is not None:
            m["clip_gain"] = m["clip_sim"] - clip_content_ref
    return m
