"""Method 2 — Fast feed-forward mood transfer with AdaIN (Huang & Belongie, ICCV 2017).

Pipeline::

    content ──► VGG enc (relu4_1) ──► f_c ─┐
                                           ├─► AdaIN ─► t ─► α·t + (1-α)·f_c ─► decoder ─► output
    mood    ──► VGG enc (relu4_1) ──► f_s ─┘

Adaptive Instance Normalisation replaces the per-channel mean/std of the content
features with those of the reference features:

    AdaIN(x, y) = σ(y) · (x − μ(x)) / σ(x) + μ(y)

Channel-wise feature statistics of a deep CNN encode "style" (colour, contrast,
texture), while the spatial arrangement of the features encodes content.

Weights: pretrained encoder + decoder from naoto0804/pytorch-AdaIN (no training done
here). See ``src/models.py`` for why the encoder must be the matching normalised VGG.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from .models import AdaINDecoder, AdaINEncoder
from .utils import (get_device, logger, np_to_tensor, pad_to_multiple, resize_max_side,
                    tensor_to_np, free_memory)

EPS = 1e-5


def calc_mean_std(feat: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel spatial mean and std of an (N,C,H,W) feature map."""
    n, c = feat.shape[:2]
    flat = feat.reshape(n, c, -1)
    mean = flat.mean(dim=2).view(n, c, 1, 1)
    std = (flat.var(dim=2, unbiased=False) + eps).sqrt().view(n, c, 1, 1)
    return mean, std


def adain_with_stats(content_feat: torch.Tensor, s_mean: torch.Tensor, s_std: torch.Tensor,
                     c_mean: torch.Tensor | None = None,
                     c_std: torch.Tensor | None = None) -> torch.Tensor:
    """AdaIN given target statistics (optionally also given the content stats, for tiling)."""
    if c_mean is None or c_std is None:
        c_mean, c_std = calc_mean_std(content_feat)
    return s_std * (content_feat - c_mean) / c_std + s_mean


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """AdaIN(x, y) = σ(y)·(x−μ(x))/σ(x) + μ(y)."""
    s_mean, s_std = calc_mean_std(style_feat)
    return adain_with_stats(content_feat, s_mean, s_std)


class AdaINTransfer:
    """Pretrained AdaIN model wrapper operating on float RGB [0,1] NumPy images."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.encoder = AdaINEncoder().to(self.device)
        self.decoder = AdaINDecoder().to(self.device)

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, img: np.ndarray) -> torch.Tensor:
        """relu4_1 features of an image (padded to a multiple of 8)."""
        x, _ = pad_to_multiple(np_to_tensor(img, self.device), 8)
        return self.encoder(x)

    @torch.no_grad()
    def mood_stats(self, refs: Sequence[np.ndarray], weights: Sequence[float] | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target (μ, σ) at relu4_1 — a weighted mix when several references are given.

        Multi-reference blending: because AdaIN is affine in (μ, σ), mixing the
        statistics linearly equals mixing the AdaIN outputs linearly (Huang &
        Belongie §7, "style interpolation").
        """
        if weights is None:
            weights = [1.0] * len(refs)
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(refs) or w.sum() <= 0 or (w < 0).any():
            raise ValueError("weights must be non-negative, one per reference, and not all zero")
        w = w / w.sum()
        mean = std = None
        for ref, wi in zip(refs, w):
            m, s = calc_mean_std(self.encode(ref))
            mean = m * wi if mean is None else mean + m * wi
            std = s * wi if std is None else std + s * wi
        return mean, std

    @torch.no_grad()
    def stylize(self, content: np.ndarray, refs: np.ndarray | Sequence[np.ndarray],
                alpha: float = 1.0, size: int | None = 512, weights: Sequence[float] | None = None,
                tile: int | None = None) -> np.ndarray:
        """Re-render ``content`` with the feature statistics of ``refs``.

        Args:
            content: float RGB [0,1].
            refs: one reference or a list (multi-reference blending).
            alpha: 0 = decoder reconstruction of the content, 1 = full AdaIN.
            size: long-side working resolution (None = keep input size).
            weights: per-reference blending weights.
            tile: if set and the image is larger than this, decode in overlapping
                tiles with *global* statistics (memory-safe, seam-free).
        Returns:
            float RGB [0,1] array of the (resized) content's shape.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if isinstance(refs, np.ndarray):
            refs = [refs]
        if size is not None:
            content = resize_max_side(content, size)
            refs = [resize_max_side(r, size) for r in refs]
        s_mean, s_std = self.mood_stats(refs, weights)
        h, w = content.shape[:2]
        if tile is not None and max(h, w) > tile:
            out = self._stylize_tiled(content, s_mean, s_std, alpha, tile)
        else:
            x, (h0, w0) = pad_to_multiple(np_to_tensor(content, self.device), 8)
            f_c = self.encoder(x)
            t = adain_with_stats(f_c, s_mean, s_std)
            t = alpha * t + (1.0 - alpha) * f_c          # content-style trade-off
            out = tensor_to_np(self.decoder(t)[..., :h0, :w0])
        free_memory(self.device)
        return out

    @torch.no_grad()
    def _stylize_tiled(self, content: np.ndarray, s_mean: torch.Tensor, s_std: torch.Tensor,
                       alpha: float, tile: int, overlap: int = 64) -> np.ndarray:
        """Tile-wise decode using content statistics of the whole image (no seams)."""
        c_mean, c_std = calc_mean_std(self.encode(resize_max_side(content, tile)))
        h, w = content.shape[:2]
        acc = np.zeros_like(content)
        wsum = np.zeros(content.shape[:2] + (1,), np.float32)
        step = tile - overlap
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        if ys[-1] + tile < h:
            ys.append(h - tile)
        if xs[-1] + tile < w:
            xs.append(w - tile)
        for y in ys:
            for x0 in xs:
                patch = content[y:y + tile, x0:x0 + tile]
                xt, (ph, pw) = pad_to_multiple(np_to_tensor(patch, self.device), 8)
                f_c = self.encoder(xt)
                t = alpha * adain_with_stats(f_c, s_mean, s_std, c_mean, c_std) + (1 - alpha) * f_c
                out = tensor_to_np(self.decoder(t)[..., :ph, :pw])
                # feathered (pyramid) weights to blend overlaps smoothly
                wy = np.minimum(np.arange(ph) + 1, np.arange(ph)[::-1] + 1).astype(np.float32)
                wx = np.minimum(np.arange(pw) + 1, np.arange(pw)[::-1] + 1).astype(np.float32)
                wgt = np.minimum.outer(np.minimum(wy, overlap), np.minimum(wx, overlap))[..., None]
                acc[y:y + ph, x0:x0 + pw] += out * wgt
                wsum[y:y + ph, x0:x0 + pw] += wgt
        return np.clip(acc / np.maximum(wsum, 1e-8), 0, 1).astype(np.float32)


_MODEL_CACHE: dict[str, AdaINTransfer] = {}


def get_adain(device: torch.device | str | None = None) -> AdaINTransfer:
    """Cached singleton per device (weights are loaded once)."""
    dev = str(torch.device(device) if device is not None else get_device())
    if dev not in _MODEL_CACHE:
        t0 = time.time()
        _MODEL_CACHE[dev] = AdaINTransfer(dev)
        logger.info("Loaded AdaIN encoder/decoder on %s in %.1fs", dev, time.time() - t0)
    return _MODEL_CACHE[dev]
